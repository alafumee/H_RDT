#!/usr/bin/env python3
"""
EgoDex dataset loader
Implements 48-dimensional hand action representation, single-view images and language embedding data loading
"""

import h5py
import numpy as np
import torch
import os
import cv2
from pathlib import Path
import warnings
import random
import json
warnings.filterwarnings("ignore")


class EgoDexDataset:
    """EgoDex dataset loader"""
    
    def __init__(self, 
                 data_root=None, 
                 config=None,
                 upsample_rate=3,
                 val=False,
                 use_precomp_lang_embed=True,
                 stat_path=None,
                 action_mode='48d'):
        """
        Args:
            data_root: Data root directory (e.g., "/share/hongzhe/datasets/egodex")
            config: Configuration dictionary
            upsample_rate: Temporal data upsampling rate (frame sampling interval)
            val: Whether it's validation set (True for test, False for train)
            use_precomp_lang_embed: Whether to use precomputed language embeddings
            stat_path: Statistics file path (default: datasets/pretrain/egodex_stat.json)
        """
        self.DATASET_NAME = "egodex_" + action_mode
        self.data_root = Path(data_root)
        self.config = config
        self.upsample_rate = upsample_rate
        self.val = val
        self.use_precomp_lang_embed = use_precomp_lang_embed

        self.action_mode = action_mode
        
        if config:
            self.chunk_size = config['common']['action_chunk_size']
            self.state_dim = config['common']['action_dim']
            self.img_history_size = config['common']['img_history_size']
        else:
            self.chunk_size = 16
            self.state_dim = 48
            self.img_history_size = 1
        
        # Set default stat_path if not provided (relative to this file)
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'egodex_stat.json')
        
        # Load data file list
        self.data_files = self._load_file_list()
        split_name = "test" if self.val else "train"
        print(f"Loaded {len(self.data_files)} {split_name} data files")
        
        # Load statistics for normalization
        self.action_min = None
        self.action_max = None
        if os.path.exists(stat_path):
            with open(stat_path, 'r') as f:
                stat = json.load(f)
            if 'egodex' in stat:
                if self.action_mode == '48d':
                    self.action_min = np.array(stat['egodex']['min'])
                    self.action_max = np.array(stat['egodex']['max'])
                elif self.action_mode == 'eef':
                    self.action_min = np.array(stat['egodex']['eef_min'])
                    self.action_max = np.array(stat['egodex']['eef_max'])
                elif self.action_mode == 'eef_rotmat':
                    self.action_min = np.array(stat['egodex']['eef_rotmat_min'])
                    self.action_max = np.array(stat['egodex']['eef_rotmat_max'])
                else:
                    raise ValueError(f"Unknown action_mode: {self.action_mode}")
                
                # self.action_min = self.action_eef_min
                # self.action_max = self.action_eef_max
    
    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME
    
    def _load_file_list(self):
        """Load data file list"""
        data_files = []
        
        if not self.val:
            # Training set: part1-part5 + extra
            for part in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra']:
                part_dir = self.data_root / part
                if part_dir.exists():
                    data_files.extend(self._scan_directory(part_dir))
        else:
            # Test set: test
            test_dir = self.data_root / 'test'
            if test_dir.exists():
                data_files.extend(self._scan_directory(test_dir))
        
        return data_files
    
    def _scan_directory(self, directory):
        """Scan files in directory"""

        # TEMPORARY: the following files' embeddings are corrupted, skip them
        corrupted_files = {'/scratch/yz12129/egodex/part1/add_remove_lid/2551.pt',
                           '/scratch/yz12129/egodex/part1/charge_uncharge_airpods/551.pt',
                           '/scratch/yz12129/egodex/part2/assemble_disassemble_furniture_bench_chair/7630.pt',
                           '/scratch/yz12129/egodex/part3/insert_remove_usb/75.pt'}
        files = []
        for task_dir in directory.iterdir():
            if task_dir.is_dir():
                # Collect all triplets: hdf5, mp4, pt
                hdf5_files = list(task_dir.glob('*.hdf5'))
                for hdf5_file in hdf5_files:
                    file_index = hdf5_file.stem  # Get filename without extension
                    mp4_file = task_dir / f"{file_index}.mp4"
                    pt_file = task_dir / f"{file_index}.pt"
                    if str(pt_file) in corrupted_files:
                        print(f"Skipping corrupted file: {pt_file}")
                        continue  
                    # Ensure all required files exist
                    if (hdf5_file.exists() and mp4_file.exists() and 
                        pt_file.exists()):
                        files.append({
                            'hdf5': hdf5_file,
                            'mp4': mp4_file,
                            'pt': pt_file,

                            'task': task_dir.name,
                            'file_index': file_index
                        })
        return files
    
    def construct_48d_action(self, hdf5_file, frame_indices):
        """
        Directly extract precomputed 48-dimensional hand action representation
        
        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract
            
        Returns:
            actions: (T, 48) action sequence
        """
        if 'actions_48d' not in hdf5_file:
            raise ValueError("Missing precomputed actions_48d data in HDF5 file, please run precompute_48d_actions.py first")
        
        # Directly read precomputed 48-dimensional action data
        precomputed_actions = hdf5_file['actions_48d'][:]
        
        # Extract actions for specified frames
        selected_actions = precomputed_actions[frame_indices]
        
        return selected_actions.astype(np.float32)
    
    def construct_eef_action(self, hdf5_file, frame_indices):
        """
        Construct 48-dimensional hand action representation from end-effector data
        
        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract
            
        Returns:
            actions: (T, 14) action sequence
        """
        if "actions_eef" not in hdf5_file:
            raise ValueError("Missing actions_eef data in HDF5 file")
        
        precomputed_actions = hdf5_file['actions_eef'][:]

        # Extract actions for specified frames
        selected_actions = precomputed_actions[frame_indices]

        return selected_actions.astype(np.float32)
    
    def construct_eef_rotmat_action(self, hdf5_file, frame_indices):
        """
        Construct 20-dimensional hand action representation with rotation matrix from end-effector data
        
        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract
            
        Returns:
            actions: (T, 20) action sequence
        """
        if "actions_eef_rotmat" not in hdf5_file:
            raise ValueError("Missing actions_eef_rotmat data in HDF5 file")
        
        precomputed_actions = hdf5_file['actions_eef_rotmat'][:]

        # Extract actions for specified frames
        selected_actions = precomputed_actions[frame_indices]

        return selected_actions.astype(np.float32)
    
    def parse_img_data(self, mp4_path, idx, horizon=None):
        """
        Load image frames following cvpr_real_dataset.py sampling logic
        
        Args:
            mp4_path: MP4 file path
            idx: Current frame index
            
        Returns:
            frames: (img_history_size, H, W, 3) image frames
        """
        horizon = self.img_history_size if horizon is None else horizon
        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling range following cvpr_real_dataset.py logic
        start_i = max(idx - horizon * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1
        
        frames = []
        
        try:
            if horizon > 0:
                for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                    if frame_idx < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            # BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        else:
                            print(f"Warning: Not enough frames in {mp4_path}")
                            break
                    else:
                        # If frame index exceeds total frames, use last valid frame
                        print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                        break
            else:
                # if horizon < 0, we want to sample future frames for visualization
                # also, the original code seems to read frame idx-2 and predict actions from idx and after??
                # so we honor that choice here, read images starting from idx
                end_i = idx - horizon # force upsample_rate=1
                for i, frame_idx in enumerate(range(idx, end_i)):
                    if frame_idx < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            # BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        else:
                            print(f"Warning: Not enough frames in {mp4_path}")
                            break
                    else:
                        # If frame index exceeds total frames, use last valid frame
                        print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                        break
        except Exception as e:
            print(f"Error loading image frames: {e}")
        
        cap.release()
        
        # Convert to numpy array
        if frames:
            frames = np.array(frames)
        else:
            frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
        
        # Pad if necessary (following cvpr_real_dataset.py logic)
        if frames.shape[0] < horizon:
            pad_frames = np.repeat(frames[:1], horizon - frames.shape[0], axis=0)
            frames = np.concatenate([pad_frames, frames], axis=0)
        
        return frames
    
    def __len__(self):
        return len(self.data_files)
    
    def extract_camera_params(self, hdf5_file, frame_idx, chunk_size):
        """
        Extract camera intrinsics and extrinsics for a given frame index
        
        Args:
            hdf5_file: HDF5 file object
            frame_idx: Frame index
            
        Returns:
            intrinsics: (T, 3, 3) camera intrinsics
            extrinsics: (T, 4, 4) camera extrinsics
        """
        camera_group = hdf5_file['camera']
        intrinsics = camera_group['intrinsic'][:]
        intrinsics = np.tile(intrinsics, (chunk_size, 1, 1))
        transforms_group = hdf5_file['transforms']
        extrinsics = transforms_group['camera'][frame_idx:frame_idx + chunk_size]
        if extrinsics.shape[0] < chunk_size:
            last_extrinsic = extrinsics[-1:] if extrinsics.shape[0] > 0 else np.eye(4)[None, ...]
            padding = np.repeat(last_extrinsic, chunk_size - extrinsics.shape[0], axis=0)
            extrinsics = np.concatenate([extrinsics, padding], axis=0)
        return intrinsics.astype(np.float32), extrinsics.astype(np.float32)
    
    def get_item(self, idx=None):
        """
        Get a data sample
        
        Returns:
            Data dictionary containing all required fields
        """
        if idx is None:
            idx = random.randint(0, len(self.data_files) - 1)
        
        file_info = self.data_files[idx % len(self.data_files)]
        
        try:
            # Load HDF5 data
            with h5py.File(file_info['hdf5'], 'r') as f:
                # Get total number of frames
                transforms_group = f['transforms']
                total_frames = list(transforms_group.values())[0].shape[0]
                
                # Calculate random index following cvpr_real_dataset.py logic
                max_index = total_frames - 2
                if max_index < 0:
                    print(f"Warning: Not enough frames in {file_info['hdf5']}")
                    return None
                
                # Random index for sampling
                index = random.randint(0, max_index)
                
                # Construct 48-dimensional actions using current index
                if self.action_mode == '48d':
                    current_action = self.construct_48d_action(f, [index])
                elif self.action_mode == 'eef':
                    current_action = self.construct_eef_action(f, [index])
                elif self.action_mode == 'eef_rotmat':
                    current_action = self.construct_eef_rotmat_action(f, [index])
                else:
                    raise ValueError(f"Unknown action_mode: {self.action_mode}")
                # current_action = self.construct_eef_action(f, [index])
                
                # Future action sequence
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_indices = list(range(index + 1, action_end + 1, self.upsample_rate))
                
                # If not enough action frames, repeat the last one
                while len(action_indices) < self.chunk_size:
                    action_indices.append(action_indices[-1] if action_indices else index + 1)
                
                # Extract action sequence
                if self.action_mode == '48d':
                    actions = self.construct_48d_action(f, action_indices[:self.chunk_size])
                elif self.action_mode == 'eef':
                    actions = self.construct_eef_action(f, action_indices[:self.chunk_size])
                elif self.action_mode == 'eef_rotmat':
                    actions = self.construct_eef_rotmat_action(f, action_indices[:self.chunk_size])
                else:
                    raise ValueError(f"Unknown action_mode: {self.action_mode}")
                # actions = self.construct_eef_action(f, action_indices[:self.chunk_size])
                
                # If actions shape is still not correct, pad with last action
                if actions.shape[0] < self.chunk_size:
                    last_action = actions[-1:] if len(actions) > 0 else current_action
                    padding = np.repeat(last_action, self.chunk_size - actions.shape[0], axis=0)
                    actions = np.concatenate([actions, padding], axis=0)

                intrinsics, extrinsics = self.extract_camera_params(f, index, self.chunk_size)
            
            # Normalize actions
            if self.action_min is not None and self.action_max is not None:
                current_action = (current_action - self.action_min) / (self.action_max - self.action_min) * 2 - 1
                current_action = np.clip(current_action, -1, 1)
                normalized_actions = (actions - self.action_min) / (self.action_max - self.action_min) * 2 - 1
                normalized_actions = np.clip(normalized_actions, -1, 1)
            
            # Load single-view image frames using new sampling logic
            image_frames = self.parse_img_data(file_info['mp4'], index)
            
            # Take only the required history size
            image_frames = image_frames[-self.img_history_size:]
            
            # Load language embedding
            lang_embed_path = file_info['pt']

            # when not visualizing, comment this out
            full_image_frames = self.parse_img_data(file_info['mp4'], index, horizon=-self.chunk_size)
            
            return {
                'states': current_action,  # (1, 48) or (1,14)!
                'actions': normalized_actions,  # (chunk_size, 48) or (chunk_size, 14)!
                'action_norm': np.ones_like(normalized_actions),  # Action indicator
                'unnormalized_actions': actions,  # (chunk_size, 48) or (chunk_size, 14)!
                'current_images': [image_frames],  # [(img_history_size, H, W, 3)] single view
                'current_images_mask': [np.ones(self.img_history_size, dtype=bool)],  # Image mask
                'instruction': str(lang_embed_path),  # Language embedding file path
                'dataset_name': self.DATASET_NAME,
                'intrinsics': intrinsics,  # (3, 3)
                'extrinsics': extrinsics,  # (4, 4)
                'task': file_info['task'],
                'file_info': {
                    'hdf5_path': str(file_info['hdf5']),
                    'mp4_path': str(file_info['mp4']),
                    'pt_path': str(file_info['pt']),
                    'total_frames': total_frames,
                    'selected_index': index,
                    'action_indices': action_indices
                },
                # when not visualizing, comment this out
                "full_current_images": [full_image_frames],  # [(chunk_size, H, W, 3)] single view
            }
            
        except Exception as e:
            print(f"Error loading data {file_info['hdf5']}: {e}")
            return None

    def __getitem__(self, idx):
        """PyTorch Dataset interface"""
        return self.get_item(idx)


if __name__ == "__main__":
    # Test dataset
    dataset = EgoDexDataset(
        data_root="/share/hongzhe/datasets/egodex",
        val=False,
        upsample_rate=3
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading samples
    sample = dataset.get_item(0)
    print("Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
