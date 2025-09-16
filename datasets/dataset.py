import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from utils.image_corrupt import image_corrupt
from datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
from datasets.pretrain.egodex_dataset import EgoDexDataset
from datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset
import h5py
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2

class VLAVisualizeDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(
        self,
        config,
        image_transform,
        num_cameras,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        image_corrupt_severity=None,
        dataset_type=None,
        state_noise_snr=None,
        use_precomp_lang_embed=True,
        upsample_rate=None,
        val=False,
        task_name="open_laptop",
        dataset_name="test_robotwin",  # Add dataset_name parameter
    ):
        super(VLAVisualizeDataset, self).__init__()
        # self.dataset_name = "egodex"
        self.dataset_name = dataset_name
        DATASET_NAMES = {self.dataset_name}
        
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = config["common"]["img_history_size"]
        self.image_transform = image_transform   

        # Initialize dataset based on dataset_name
        if self.dataset_name.startswith("egodex"):
            self.hdf5_dataset = EgoDexDataset(
                config=config,
                upsample_rate=upsample_rate,
                val=val,
                use_precomp_lang_embed=use_precomp_lang_embed,
                # Note: override default paths if needed
                data_root="/scratch/yz12129/egodex",
                stat_path="/scratch/yz12129/hrdt_pretrain/egodex_stat.json",
                action_mode=self.dataset_name.split("egodex_")[1] if "egodex_" in self.dataset_name else self.dataset_name,  # e.g., '48d', 'eef', 'eef_rotmat'
            )
        elif self.dataset_name == "robotwin_agilex":
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="multi_task",
                config=config,
                # Note: override default paths
                multi_task_root_dir="/path/to/robotwin2",
            )
            '''
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder="Aloha-AgileX/data",
                max_episodes=50,
                config=config
                # Note: override default paths
                # single_task_root_dir="/path/to/your/robotwin2/single",
            )
            '''
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
            
        print(f"Initialized dataset: {self.dataset_name}")

        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.dataset_type = dataset_type

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        # self.image_aug_transform = get_image_augmentation()
        self.image_aug = image_aug

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        # Get data from backend dataset
        try:
            res = self.hdf5_dataset.get_item(index)
        except Exception as e:
            print(f"Error loading episode {index}: {e}", flush=True)
            return None
            
        # Add check for res being None, retry a few times if it's None
        retry_count = 0
        max_retries = 5
        while res is None and retry_count < max_retries:
            retry_count += 1
            print(f"Got None data item, retrying {retry_count} time...", flush=True)
            try:
                res = self.hdf5_dataset.get_item(index)
            except Exception as e:
                print(f"Error during retry data loading: {e}")
                
        # If still None after multiple retries, return a default value to prevent training interruption
        if res is None:
            print(f"Warning: Still unable to get valid data after multiple retries, returning default value", flush=True)

        data_dict = {}
        data_dict['dataset_name'] = res['dataset_name']
        data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]

        # Process state and action data
        data_dict["states"] = res['states']
        data_dict["actions"] = res['actions']
        data_dict["action_norm"] = res['action_norm']
        data_dict["unnormalized_actions"] = res['unnormalized_actions']

        # camera
        data_dict['intrinsics'] = res['intrinsics']
        data_dict['extrinsics'] = res['extrinsics']

        # Process images
        if self.dataset_name.startswith('egodex'):
            # Single camera / stitched image processing
            image_metas = []
            images = res['current_images'][0]
            valid_mask = res.get('current_images_mask', [np.ones(self.img_history_size, dtype=bool)])[0]
            image_metas.append((images, valid_mask))
            
            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                images, valid_mask = image_metas[0]
                if valid_mask[hist_idx]:
                    rearranged_images.append((images[hist_idx], True))
                else:
                    rearranged_images.append((None, False))
        else:
            # Multi-view processing (original logic)
            image_metas = []
            for cam_idx in range(self.num_cameras):
                images = res['current_images'][cam_idx]
                valid_mask = res.get('current_images_mask', np.ones((self.num_cameras, self.img_history_size), dtype=bool))[cam_idx]
                image_metas.append((images, valid_mask))

            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                for cam_idx in range(self.num_cameras):
                    images, valid_mask = image_metas[cam_idx]
                    if valid_mask[hist_idx]:
                        rearranged_images.append((images[hist_idx], True))
                    else:
                        rearranged_images.append((None, False))

        all_pixel_values = []
        for image, valid in rearranged_images:
            image = Image.fromarray(image) if image is not None else None

            if valid and self.auto_adjust_image_brightness:
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

            # Only apply image augmentation to 50% of the images
            if valid and self.image_aug and (random.random() > 0.5):
                aug_type = random.choice([
                    "corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)
                # image = self.image_aug_transform(image)

            pixel_values = self.image_transform(image)
            all_pixel_values.append(pixel_values)

        # Process dino-siglip format images
        pv_example = all_pixel_values[0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in all_pixel_values]
            )
            for k in pv_example
        }
        data_dict["images"] = merged_pixel_values

        data_dict["original_images"] = res['full_current_images'][0]

        if self.use_precomp_lang_embed:
            # All datasets should provide lang_embeds as tensor
            if "lang_embeds" in res:
                data_dict["lang_embeds"] = res["lang_embeds"]
            elif torch.is_tensor(res["instruction"]):
                data_dict["lang_embeds"] = res["instruction"]
            else:
                # Legacy: load from file path
                try:
                    data_dict["lang_embeds"] = torch.load(res["instruction"])["embeddings"].squeeze(0)
                except Exception as e:
                    print(f"Error loading language embeddings from {res['instruction']}: {e}", flush=True)
                    raise e
                    # data_dict["lang_embeds"] = torch.zeros((1, 1024), dtype=torch.float32)

        # Convert all numpy arrays to torch tensors
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        # Verify all data is tensors
        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

        return data_dict


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(
        self,
        config,
        image_transform,
        num_cameras,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        image_corrupt_severity=None,
        dataset_type=None,
        state_noise_snr=None,
        use_precomp_lang_embed=True,
        upsample_rate=None,
        val=False,
        task_name="open_laptop",
        dataset_name="test_robotwin",  # Add dataset_name parameter
    ):
        super(VLAConsumerDataset, self).__init__()
        # self.dataset_name = "egodex"
        self.dataset_name = dataset_name
        DATASET_NAMES = {self.dataset_name}
        
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = config["common"]["img_history_size"]
        self.image_transform = image_transform   

        # Initialize dataset based on dataset_name
        if self.dataset_name.startswith("egodex"):
            self.hdf5_dataset = EgoDexDataset(
                config=config,
                upsample_rate=upsample_rate,
                val=val,
                use_precomp_lang_embed=use_precomp_lang_embed,
                # Note: override default paths if needed
                data_root="/scratch/yz12129/egodex",
                stat_path="/scratch/yz12129/hrdt_pretrain/egodex_stat.json",
                action_mode=self.dataset_name.split("egodex_")[1] if "egodex_" in self.dataset_name else self.dataset_name,  # e.g., '48d', 'eef', 'eef_rotmat'
            )
        elif self.dataset_name == "robotwin_agilex":
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="multi_task",
                config=config,
                # Note: override default paths
                multi_task_root_dir="/scratch/yz12129/robotwin/dataset",
            )
            '''
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder="Aloha-AgileX/data",
                max_episodes=50,
                config=config
                # Note: override default paths
                # single_task_root_dir="/path/to/your/robotwin2/single",
            )
            '''
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
            
        print(f"Initialized dataset: {self.dataset_name}")

        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.dataset_type = dataset_type

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        # self.image_aug_transform = get_image_augmentation()
        self.image_aug = image_aug

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        # Get data from backend dataset
        try:
            res = self.hdf5_dataset.get_item(index)
        except Exception as e:
            print(f"Error loading episode {index}: {e}", flush=True)
            return None
            
        # Add check for res being None, retry a few times if it's None
        retry_count = 0
        max_retries = 5
        while res is None and retry_count < max_retries:
            retry_count += 1
            print(f"Got None data item, retrying {retry_count} time...", flush=True)
            try:
                res = self.hdf5_dataset.get_item(index)
            except Exception as e:
                print(f"Error during retry data loading: {e}")
                
        # If still None after multiple retries, return a default value to prevent training interruption
        if res is None:
            print(f"Warning: Still unable to get valid data after multiple retries, returning default value", flush=True)

        data_dict = {}
        data_dict['dataset_name'] = res['dataset_name']
        data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]

        # Process state and action data
        data_dict["states"] = res['states']
        data_dict["actions"] = res['actions']
        data_dict["action_norm"] = res['action_norm']

        # Process images
        if self.dataset_name.startswith('egodex'):
            # Single camera / stitched image processing
            image_metas = []
            images = res['current_images'][0]
            valid_mask = res.get('current_images_mask', [np.ones(self.img_history_size, dtype=bool)])[0]
            image_metas.append((images, valid_mask))
            
            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                images, valid_mask = image_metas[0]
                if valid_mask[hist_idx]:
                    rearranged_images.append((images[hist_idx], True))
                else:
                    rearranged_images.append((None, False))
        else:
            # Multi-view processing (original logic)
            image_metas = []
            for cam_idx in range(self.num_cameras):
                images = res['current_images'][cam_idx]
                valid_mask = res.get('current_images_mask', np.ones((self.num_cameras, self.img_history_size), dtype=bool))[cam_idx]
                image_metas.append((images, valid_mask))

            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                for cam_idx in range(self.num_cameras):
                    images, valid_mask = image_metas[cam_idx]
                    if valid_mask[hist_idx]:
                        rearranged_images.append((images[hist_idx], True))
                    else:
                        rearranged_images.append((None, False))

        all_pixel_values = []
        for image, valid in rearranged_images:
            image = Image.fromarray(image) if image is not None else None

            if valid and self.auto_adjust_image_brightness:
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

            # Only apply image augmentation to 50% of the images
            if valid and self.image_aug and (random.random() > 0.5):
                aug_type = random.choice([
                    "corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)
                # image = self.image_aug_transform(image)

            pixel_values = self.image_transform(image)
            all_pixel_values.append(pixel_values)

        # Process dino-siglip format images
        pv_example = all_pixel_values[0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in all_pixel_values]
            )
            for k in pv_example
        }
        data_dict["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            # All datasets should provide lang_embeds as tensor
            if "lang_embeds" in res:
                data_dict["lang_embeds"] = res["lang_embeds"]
            elif torch.is_tensor(res["instruction"]):
                data_dict["lang_embeds"] = res["instruction"]
            else:
                # Legacy: load from file path
                try:
                    data_dict["lang_embeds"] = torch.load(res["instruction"])["embeddings"].squeeze(0)
                except Exception as e:
                    print(f"Error loading language embeddings from {res['instruction']}: {e}", flush=True)
                    raise e
                    # data_dict["lang_embeds"] = torch.zeros((1, 1024), dtype=torch.float32)

        # Convert all numpy arrays to torch tensors
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        # Verify all data is tensors
        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

        return data_dict

class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, use_precomp_lang_embed=True) -> None:
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "images": [],
            "data_indices": [],
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # Process images
            batch["images"].append(instance["images"])
            batch["data_indices"].append(instance["data_idx"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)


        # Process dino-siglip format images
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        batch["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask

        return batch
    
class DataCollatorForVLAVisualizeDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, use_precomp_lang_embed=True) -> None:
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "unnormalized_actions": [],
            "images": [],
            "data_indices": [],
            "intrinsics": [],
            "extrinsics": [],
            "original_images": [],
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data
            keys_to_check = [
                'states', 'actions',
                'action_norm',
                'unnormalized_actions',
                'original_images',
                'intrinsics',
                'extrinsics',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                elif isinstance(instance[key], np.ndarray):
                    item = torch.from_numpy(instance[key])
                else:
                    raise ValueError(f"Unsupported data type for key {key}: {type(instance[key])}")
                batch[key].append(item)

            # Process images
            batch["images"].append(instance["images"])
            batch["data_indices"].append(instance["data_idx"])

            # batch["intrinsics"].append(instance["intrinsics"])
            # batch["extrinsics"].append(instance["extrinsics"])

            # batch["original_images"].append(instance["original_images"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
            'unnormalized_actions',
            'original_images',
            'intrinsics',
            'extrinsics',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # Process dino-siglip format images
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        batch["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask

        return batch