from collections import defaultdict

import torch
import torch.nn.functional as F

from .renderer import EgoDexRenderer

# def visualize_sample_res(images, pred_actions):
#     pass

@torch.no_grad()
def log_sample_res(
    hrdt, args, config, accelerator, weight_dtype, dataset_id2name, 
    dataloader, logger, vision_encoder,
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )
    hrdt.eval()

    visualize = args.visualize_during_training

    # vis_prob = 0.01

    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    # Initialize overall counters
    loss_counter["overall_avg_sample_mse"] = 0
    loss_counter["overall_avg_sample_l2err"] = 0

    renderer = EgoDexRenderer()

    import json, numpy as np
    from datetime import datetime
    with open('/scratch/yz12129/hrdt_pretrain/egodex_stat.json', 'r') as f:
        stat = json.load(f)
    # action_min = np.array(stat['egodex']['eef_rotmat_min'])
    # action_max = np.array(stat['egodex']['eef_rotmat_max'])
    # print("action_rotmat_min:", action_min.shape)
    # print("action_rotmat_max:", action_max.shape)
    # action_eef_min = np.array(stat['egodex']['eef_min'])
    # action_eef_max = np.array(stat['egodex']['eef_max'])
    # print("action_eef_min:", action_eef_min.shape)
    # print("action_eef_max:", action_eef_max.shape)
    if args.action_mode == '48d':
        action_min = np.array(stat['egodex']['min'])
        action_max = np.array(stat['egodex']['max'])
    elif args.action_mode == 'eef_rotmat':
        action_min = np.array(stat['egodex']['eef_rotmat_min'])
        action_max = np.array(stat['egodex']['eef_rotmat_max'])
    elif args.action_mode == 'eef':
        action_min = np.array(stat['egodex']['eef_min'])
        action_max = np.array(stat['egodex']['eef_max'])

    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break
        
        if visualize:
            original_image_sequence = batch['original_images'][0].cpu().numpy() # (T, H, W, 3), uint8
            renderer.consume_data(image_frames=original_image_sequence)

            extrinsics = batch['extrinsics'][0].cpu().numpy() # (T, 4, 4)
            intrinsics = batch['intrinsics'][0].cpu().numpy() # (T, 3, 3)
            renderer.consume_data(intrinsics=intrinsics, extrinsics=extrinsics)

        # Process image data
        if isinstance(batch["images"], dict):
            # {"dino": (B, T, C, H, W), "dino": (B, T, C, H, W)}
            images = {k: v.to(dtype=weight_dtype) for k, v in batch["images"].items()}
        else:
            raise ValueError(f"Unsupported `batch[\"images\"]` type = {type(batch['images'])}")

        # Extract VLM features
        with torch.no_grad():
            k = next(iter(images))
            batch_size, _, C, H, W = images[k].shape
            for k in images:
                images[k] = images[k].view(-1, C, H, W)
            image_features = vision_encoder(images).detach()
            image_features = image_features.view((batch_size, -1, vision_encoder.embed_dim))

        # Process language data based on training mode
        lang_embeds = None
        lang_attn_mask = None
        if args.training_mode == "lang":
            lang_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
            lang_attn_mask = batch["lang_attn_mask"].to(dtype=weight_dtype)

        # Get current state
        states = batch["states"].to(dtype=weight_dtype)
        
        # Get ground truth actions for evaluation
        actions = batch["actions"].to(weight_dtype)
        action_norm = batch["action_norm"].to(weight_dtype)
        dataset_indices = batch["data_indices"]
        
        # Sample actions using the model
        pred_actions = hrdt.predict_action(
            state_tokens=states,
            image_tokens=image_features,
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
        )

        if visualize and accelerator.is_main_process:

            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if args.action_mode == '48d':

                # left hand

                scaled_pred_actions = (pred_actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min
                scaled_gt_actions = (actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min

                print("gt 48d action:", scaled_gt_actions[0, :, :9])
                # comment this for pred
                # scaled_pred_actions = scaled_gt_actions.copy()

                pred_first_rot = scaled_pred_actions[0, :, 3:9]
                pred_rot_x = pred_first_rot[:, :3] # first column of rotation matrix
                orthonormalized_x = pred_rot_x / np.linalg.norm(pred_rot_x, axis=1, keepdims=True)
                pred_rot_y = pred_first_rot[:, 3:6] # second column of rotation matrix
                orthonormalized_y = pred_rot_y - (np.sum(pred_rot_y * orthonormalized_x, axis=1, keepdims=True) * orthonormalized_x)
                orthonormalized_y = orthonormalized_y / np.linalg.norm(orthonormalized_y, axis=1, keepdims=True)
                ortho_pred_rot_x = torch.tensor(orthonormalized_x, dtype=torch.float32)
                ortho_pred_rot_y = torch.tensor(orthonormalized_y, dtype=torch.float32)
                ortho_pred_rot_z = torch.cross(torch.tensor(ortho_pred_rot_x), torch.tensor(ortho_pred_rot_y), dim=1)
                rot_mat = torch.stack([ortho_pred_rot_x, ortho_pred_rot_y, ortho_pred_rot_z], dim=2) # (num_steps, 3, 3)
                rot_mat = rot_mat.cpu().numpy()

                # for right hand
                pred_first_rot_right = scaled_pred_actions[0, :, 27:33]
                pred_rot_x_right = pred_first_rot_right[:, :3] # first column of rotation
                orthonormalized_x_right = pred_rot_x_right / np.linalg.norm(pred_rot_x_right, axis=1, keepdims=True)
                pred_rot_y_right = pred_first_rot_right[:, 3:6] # second column
                orthonormalized_y_right = pred_rot_y_right - (np.sum(pred_rot_y_right * orthonormalized_x_right, axis=1, keepdims=True) * orthonormalized_x_right)
                orthonormalized_y_right = orthonormalized_y_right / np.linalg.norm(orthonormalized_y_right, axis=1, keepdims=True)
                ortho_pred_rot_x_right = torch.tensor(orthonormalized_x_right, dtype=torch.float32)
                ortho_pred_rot_y_right = torch.tensor(orthonormalized_y_right, dtype=torch.float32)
                ortho_pred_rot_z_right = torch.cross(torch.tensor(ortho_pred_rot_x_right), torch.tensor(ortho_pred_rot_y_right), dim=1)
                rot_mat_right = torch.stack([ortho_pred_rot_x_right, ortho_pred_rot_y_right, ortho_pred_rot_z_right], dim=2) # (num_steps, 3, 3)
                rot_mat_right = rot_mat_right.cpu().numpy()

                renderer.consume_data(eef_pos=scaled_pred_actions[0, :, :3], eef_rot=rot_mat,
                                    eef2_pos=scaled_pred_actions[0, :, 24:27], eef2_rot=rot_mat_right)
                
                # repeat for ground truth
                gt_first_rot = scaled_gt_actions[0, :, 3:9]
                gt_rot_x = gt_first_rot[:, :3] # first column of rotation matrix
                orthonormalized_x = gt_rot_x / np.linalg.norm(gt_rot_x, axis=1, keepdims=True)
                gt_rot_y = gt_first_rot[:, 3:6] # second column of rotation matrix
                orthonormalized_y = gt_rot_y - (np.sum(gt_rot_y * orthonormalized_x, axis=1, keepdims=True) * orthonormalized_x)
                orthonormalized_y = orthonormalized_y / np.linalg.norm(orthonormalized_y, axis=1, keepdims=True)
                ortho_gt_rot_x = torch.tensor(orthonormalized_x, dtype=torch.float32)
                ortho_gt_rot_y = torch.tensor(orthonormalized_y, dtype=torch.float32)
                ortho_gt_rot_z = torch.cross(torch.tensor(ortho_gt_rot_x), torch.tensor(ortho_gt_rot_y), dim=1)
                rot_mat = torch.stack([ortho_gt_rot_x, ortho_gt_rot_y, ortho_gt_rot_z], dim=2) # (num_steps, 3, 3)
                rot_mat = rot_mat.cpu().numpy()

                # for right hand
                gt_first_rot_right = scaled_gt_actions[0, :, 27:33]
                gt_rot_x_right = gt_first_rot_right[:, :3] # first column of rotation
                orthonormalized_x_right = gt_rot_x_right / np.linalg.norm(gt_rot_x_right, axis=1, keepdims=True)
                gt_rot_y_right = gt_first_rot_right[:, 3:6] # second column
                orthonormalized_y_right = gt_rot_y_right - (np.sum(gt_rot_y_right * orthonormalized_x_right, axis=1, keepdims=True) * orthonormalized_x_right)
                orthonormalized_y_right = orthonormalized_y_right / np.linalg.norm(orthonormalized_y_right, axis=1, keepdims=True)
                ortho_gt_rot_x_right = torch.tensor(orthonormalized_x_right, dtype=torch.float32)
                ortho_gt_rot_y_right = torch.tensor(orthonormalized_y_right, dtype=torch.float32)
                ortho_gt_rot_z_right = torch.cross(torch.tensor(ortho_gt_rot_x_right), torch.tensor(ortho_gt_rot_y_right), dim=1)
                rot_mat_right = torch.stack([ortho_gt_rot_x_right, ortho_gt_rot_y_right, ortho_gt_rot_z_right], dim=2) # (num_steps, 3, 3)
                rot_mat_right = rot_mat_right.cpu().numpy()

                renderer.consume_data(gt_eef_pos=scaled_gt_actions[0, :, :3], gt_eef_rot=rot_mat,
                                    gt_eef2_pos=scaled_gt_actions[0, :, 24:27], gt_eef2_rot=rot_mat_right)
                
                renderer.run(output_path=f'/home/yz12129/hrdt_vis/sample_vis/48d_{now_time}_pred.avi')
            
            elif args.action_mode == 'eef_rotmat':

                # left hand

                scaled_pred_actions = (pred_actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min
                scaled_gt_actions = (actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min

                # comment this for pred
                # scaled_pred_actions = scaled_gt_actions.copy()

                pred_first_rot = scaled_pred_actions[0, :, 3:9]
                pred_rot_x = pred_first_rot[:, :3] # first column of rotation matrix
                orthonormalized_x = pred_rot_x / np.linalg.norm(pred_rot_x, axis=1, keepdims=True)
                pred_rot_y = pred_first_rot[:, 3:6] # second column of rotation matrix
                orthonormalized_y = pred_rot_y - (np.sum(pred_rot_y * orthonormalized_x, axis=1, keepdims=True) * orthonormalized_x)
                orthonormalized_y = orthonormalized_y / np.linalg.norm(orthonormalized_y, axis=1, keepdims=True)
                ortho_pred_rot_x = torch.tensor(orthonormalized_x, dtype=torch.float32)
                ortho_pred_rot_y = torch.tensor(orthonormalized_y, dtype=torch.float32)
                ortho_pred_rot_z = torch.cross(torch.tensor(ortho_pred_rot_x), torch.tensor(ortho_pred_rot_y), dim=1)
                rot_mat = torch.stack([ortho_pred_rot_x, ortho_pred_rot_y, ortho_pred_rot_z], dim=2) # (num_steps, 3, 3)
                rot_mat = rot_mat.cpu().numpy()

                # for right hand
                pred_first_rot_right = scaled_pred_actions[0, :, 13:19]
                pred_rot_x_right = pred_first_rot_right[:, :3] # first column of rotation
                orthonormalized_x_right = pred_rot_x_right / np.linalg.norm(pred_rot_x_right, axis=1, keepdims=True)
                pred_rot_y_right = pred_first_rot_right[:, 3:6] # second column
                orthonormalized_y_right = pred_rot_y_right - (np.sum(pred_rot_y_right * orthonormalized_x_right, axis=1, keepdims=True) * orthonormalized_x_right)
                orthonormalized_y_right = orthonormalized_y_right / np.linalg.norm(orthonormalized_y_right, axis=1, keepdims=True)
                ortho_pred_rot_x_right = torch.tensor(orthonormalized_x_right, dtype=torch.float32)
                ortho_pred_rot_y_right = torch.tensor(orthonormalized_y_right, dtype=torch.float32)
                ortho_pred_rot_z_right = torch.cross(torch.tensor(ortho_pred_rot_x_right), torch.tensor(ortho_pred_rot_y_right), dim=1)
                rot_mat_right = torch.stack([ortho_pred_rot_x_right, ortho_pred_rot_y_right, ortho_pred_rot_z_right], dim=2) # (num_steps, 3, 3)
                rot_mat_right = rot_mat_right.cpu().numpy()

                renderer.consume_data(eef_pos=scaled_pred_actions[0, :, :3], eef_rot=rot_mat,
                                    eef2_pos=scaled_pred_actions[0, :, 10:13], eef2_rot=rot_mat_right)
                
                # repeat for ground truth
                gt_first_rot = scaled_gt_actions[0, :, 3:9]
                gt_rot_x = gt_first_rot[:, :3] # first column of rotation matrix
                orthonormalized_x = gt_rot_x / np.linalg.norm(gt_rot_x, axis=1, keepdims=True)
                gt_rot_y = gt_first_rot[:, 3:6] # second column of rotation matrix
                orthonormalized_y = gt_rot_y - (np.sum(gt_rot_y * orthonormalized_x, axis=1, keepdims=True) * orthonormalized_x)
                orthonormalized_y = orthonormalized_y / np.linalg.norm(orthonormalized_y, axis=1, keepdims=True)
                ortho_gt_rot_x = torch.tensor(orthonormalized_x, dtype=torch.float32)
                ortho_gt_rot_y = torch.tensor(orthonormalized_y, dtype=torch.float32)
                ortho_gt_rot_z = torch.cross(torch.tensor(ortho_gt_rot_x), torch.tensor(ortho_gt_rot_y), dim=1)
                rot_mat = torch.stack([ortho_gt_rot_x, ortho_gt_rot_y, ortho_gt_rot_z], dim=2) # (num_steps, 3, 3)
                rot_mat = rot_mat.cpu().numpy()

                # for right hand
                gt_first_rot_right = scaled_gt_actions[0, :, 13:19]
                gt_rot_x_right = gt_first_rot_right[:, :3] # first column of rotation
                orthonormalized_x_right = gt_rot_x_right / np.linalg.norm(gt_rot_x_right, axis=1, keepdims=True)
                gt_rot_y_right = gt_first_rot_right[:, 3:6] # second column
                orthonormalized_y_right = gt_rot_y_right - (np.sum(gt_rot_y_right * orthonormalized_x_right, axis=1, keepdims=True) * orthonormalized_x_right)
                orthonormalized_y_right = orthonormalized_y_right / np.linalg.norm(orthonormalized_y_right, axis=1, keepdims=True)
                ortho_gt_rot_x_right = torch.tensor(orthonormalized_x_right, dtype=torch.float32)
                ortho_gt_rot_y_right = torch.tensor(orthonormalized_y_right, dtype=torch.float32)
                ortho_gt_rot_z_right = torch.cross(torch.tensor(ortho_gt_rot_x_right), torch.tensor(ortho_gt_rot_y_right), dim=1)
                rot_mat_right = torch.stack([ortho_gt_rot_x_right, ortho_gt_rot_y_right, ortho_gt_rot_z_right], dim=2) # (num_steps, 3, 3)
                rot_mat_right = rot_mat_right.cpu().numpy()

                renderer.consume_data(gt_eef_pos=scaled_gt_actions[0, :, :3], gt_eef_rot=rot_mat,
                                    gt_eef2_pos=scaled_gt_actions[0, :, 10:13], gt_eef2_rot=rot_mat_right)
                
                renderer.run(output_path=f'/home/yz12129/hrdt_vis/sample_vis/eef_rotmat_{now_time}_pred.avi')

            elif args.action_mode == 'eef':

                scaled_pred_actions = (pred_actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min
                scaled_gt_actions = (actions.cpu().to(torch.float32).numpy() + 1) / 2 * (action_max - action_min) + action_min

                from scipy.spatial.transform import Rotation as R
                pred_eef_rot = R.from_rotvec(scaled_pred_actions[0, :, 3:6]).as_matrix() # (num_steps, 3, 3)
                pred_eef2_rot = R.from_rotvec(scaled_pred_actions[0, :, 10:13]).as_matrix() # (num_steps, 3, 3)
                renderer.consume_data(eef_pos=scaled_pred_actions[0, :, :3], eef_rot=pred_eef_rot,
                                    eef2_pos=scaled_pred_actions[0, :, 7:10], eef2_rot=pred_eef2_rot)
                
                gt_eef_rot = R.from_rotvec(scaled_gt_actions[0, :, 3:6]).as_matrix() # (num_steps, 3, 3)
                gt_eef2_rot = R.from_rotvec(scaled_gt_actions[0, :, 10:13]).as_matrix() # (num_steps, 3, 3)
                renderer.consume_data(gt_eef_pos=scaled_gt_actions[0, :, :3], gt_eef_rot=gt_eef_rot,
                                    gt_eef2_pos=scaled_gt_actions[0, :, 7:10], gt_eef2_rot=gt_eef2_rot)
                renderer.run(output_path=f'/home/yz12129/hrdt_vis/sample_vis/eef_{now_time}_pred.avi')
                


        num_steps = pred_actions.shape[1]
        expanded_action_norm = action_norm.float()

        # Compute metrics
        loss = F.mse_loss(pred_actions, actions, reduction='none').float()

        batch_size = pred_actions.shape[0]
        mse_loss_per_entry = loss.reshape((batch_size, -1)).mean(1)
        l2_loss_per_entry = loss.sqrt() / (expanded_action_norm + 1e-3)
        l2_loss_per_entry = l2_loss_per_entry.reshape((batch_size, -1)).mean(1)

        # Gather metrics across processes
        dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics(
            (torch.LongTensor(dataset_indices).to(device=pred_actions.device),
             mse_loss_per_entry, l2_loss_per_entry),
        )
        dataset_indices = dataset_indices.tolist()
        
        mse_loss_all = mse_losses
        overall_mse = mse_loss_all.mean().item()
        loss_for_log["overall_avg_sample_mse"] += overall_mse

        l2_loss_all = l2_losses
        overall_l2 = l2_loss_all.mean().item()
        loss_for_log["overall_avg_sample_l2err"] += overall_l2

        # Log metrics per dataset
        if accelerator.is_main_process:
            for loss_suffix, losses in zip(["_sample_mse", "_sample_l2err"], [mse_losses, l2_losses]):
                for dataset_idx, loss_tensor in zip(dataset_indices, losses):
                    loss_name = dataset_id2name[dataset_idx] + loss_suffix
                    loss_for_log[loss_name] += loss_tensor.item()
                    loss_counter[loss_name] += 1

        # Increment overall counters
        loss_counter["overall_avg_sample_mse"] += 1
        loss_counter["overall_avg_sample_l2err"] += 1

    # Average metrics
    for name in loss_for_log:
        loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)

    result_dict = {}
    for name, value in dict(loss_for_log).items():
        if name.startswith("overall_avg_"):
            new_name = name.replace("overall_avg_sample_", "overall_avg_")
            result_dict[f"action/metrics/{new_name}"] = value
        else:
            new_name = name.replace("_sample_", "_")
            result_dict[f"action/dataset_metrics/{new_name}"] = value

    hrdt.train()
    torch.cuda.empty_cache()

    return result_dict