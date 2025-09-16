class EgoDexRenderer:
    def __init__(self):
        self.image_frames = None  # List of image frames (T, H, W, 3)
        self.eef_pos = None       # End-effector positions (T, 3)
        self.eef_rot = None       # End-effector rotations (T, 4) as quaternions
        self.extrinsics = None    # Camera extrinsics (T, 4, 4)
        self.intrinsics = None    # Camera intrinsics (T, 3, 3)
        self.eef2_pos = None  # For bimanual
        self.eef2_rot = None  # For bimanual
        self.gt_eef_pos = None
        self.gt_eef_rot = None
        self.gt_eef2_pos = None
        self.gt_eef2_rot = None
        self._debug_frame_idx = 0

    def consume_data(self, image_frames=None, eef_pos=None, eef_rot=None, extrinsics=None, intrinsics=None,
                    eef2_pos=None, eef2_rot=None, gt_eef_pos=None, gt_eef_rot=None, gt_eef2_pos=None, gt_eef2_rot=None):
        """
        :param image_frames: List of image frames (T, H, W, 3)
        :param eef_pos: End-effector positions (T, 3)
        :param eef_rot: End-effector rotations (T, 3, 3) as rotation matrices
        :param extrinsics: Camera extrinsics (T, 4, 4)
        :param intrinsics: Camera intrinsics (T, 3, 3)
        """
        if image_frames is not None:
            self.image_frames = image_frames
        if eef_pos is not None:
            self.eef_pos = eef_pos
        if eef_rot is not None:
            self.eef_rot = eef_rot
        if extrinsics is not None:
            self.extrinsics = extrinsics
        if intrinsics is not None:
            self.intrinsics = intrinsics
        if eef2_pos is not None:
            self.eef2_pos = eef2_pos
        if eef2_rot is not None:
            self.eef2_rot = eef2_rot
        if gt_eef_pos is not None:
            self.gt_eef_pos = gt_eef_pos
        if gt_eef_rot is not None:
            self.gt_eef_rot = gt_eef_rot
        if gt_eef2_pos is not None:
            self.gt_eef2_pos = gt_eef2_pos
        if gt_eef2_rot is not None:
            self.gt_eef2_rot = gt_eef2_rot

    def transform_world_to_pixel(self, points, extrinsics, intrinsics):
        """
        Transform 3D world points to 2D pixel coordinates.
        :param points: (N, 3) array of 3D points in world coordinates
        :param extrinsics: (4, 4) camera extrinsics matrix
        :param intrinsics: (3, 3) camera intrinsics matrix
        :return: (N, 2) array of 2D pixel coordinates
        """
        import numpy as np
        # Convert points to homogeneous coordinates
        num_points = points.shape[0]
        points_homogeneous = np.hstack((points, np.ones((num_points, 1))))  # (N, 4)

        camera_rotation = extrinsics[:3, :3]
        camera_translation = extrinsics[:3, 3]
        world_to_camera_transform = np.eye(4)
        world_to_camera_transform[:3, :3] = camera_rotation.T
        world_to_camera_transform[:3, 3] = -camera_rotation.T @ camera_translation
        
        points_camera = (world_to_camera_transform @ points_homogeneous.T).T  # (N, 4)
        points_camera = points_camera[:, :3]  # (N, 3)
        # points_camera = points_camera[points_camera[:, 2] > 0]  # Keep points in front of the camera
        points_2d_homogeneous = (intrinsics @ points_camera.T).T  # (N, 3)
        pixel_coords = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # (N, 2)

        return pixel_coords

    def render_eef_on_image(self, image, eef_pos, eef_rot, intrinsics, extrinsics):
        """
        Render the end-effector on a single image frame.
        :param image: Single image frame (H, W, 3)
        :param eef_pos: End-effector position (3,)
        :param eef_rot: End-effector rotation (3, 3) as rotation matrix
        :param intrinsics: Camera intrinsics (3, 3)
        :param extrinsics: Camera extrinsics (4, 4)
        :return: Image with rendered end-effector
        """
        import cv2
        import numpy as np
        import os
        # Project the 3D model of the end-effector
        # onto the 2D image plane using the provided camera parameters.
        rendered_image = image.copy()
        rendered_image = rendered_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        rendered_image = np.ascontiguousarray(rendered_image, dtype=np.uint8)
        # Example: Draw a simple circle at the projected position of the end-effector
        
        # from scipy.spatial.transform import Rotation as R
        save_dir = "/home/yz12129/tmp/eef_render_debug"
        os.makedirs(save_dir, exist_ok=True)
        frame_idx = getattr(self, "_debug_frame_idx", 0)
        # cv2.imwrite(os.path.join(save_dir, f"frame_orig_{frame_idx:04d}.png"), rendered_image)
        
        # Project the 3D position to 2D (this is a simplified example)
        projected_point = self.transform_world_to_pixel(eef_pos[None, :], extrinsics, intrinsics)[0]

        x, y = int(projected_point[0]), int(projected_point[1])
        print("projected eef position:", x, y, flush=True)
        cv2.circle(rendered_image, (x, y), 5, (0, 255, 0), -1)  # Draw green circle

        # cv2.imwrite(os.path.join(save_dir, f"frame_pt_{frame_idx:04d}.png"), rendered_image)
        
        # render eef orientation
        # Draw orientation axes

        # Define axis length in meters
        axis_length = 0.05

        # Define axis endpoints in eef frame
        axes = np.eye(3) * axis_length  # (3, 3)
        axes_points = eef_pos[None, :] + (eef_rot @ axes.T).T  # (3, 3)

        # Project each axis endpoint to 2D
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X=red, Y=green, Z=blue (BGR)
            proj_axis = self.transform_world_to_pixel(axes_points[i:i+1, :], extrinsics, intrinsics)[0]
            x2, y2 = int(proj_axis[0]), int(proj_axis[1])
            cv2.line(rendered_image, (x, y), (x2, y2), color, 2)

        # cv2.imwrite(os.path.join(save_dir, f"frame_axes_{frame_idx:04d}.png"), rendered_image)
        self._debug_frame_idx = frame_idx + 1

        return rendered_image
    
    def render_bimanual_eef_on_image(self, image, eef1_pos, eef1_rot, eef2_pos, eef2_rot, intrinsics, extrinsics):
        """
        Render the end-effectors on a single image frame.
        :param image: Single image frame (H, W, 3)
        :param eef1_pos: End-effector 1 position (3,)
        :param eef1_rot: End-effector 1 rotation (3, 3) as rotation matrix
        :param eef2_pos: End-effector 2 position (3,)
        :param eef2_rot: End-effector 2 rotation (3, 3) as rotation matrix
        :param intrinsics: Camera intrinsics (3, 3)
        :param extrinsics: Camera extrinsics (4, 4)
        :return: Image with rendered end-effectors
        """
        import cv2
        import numpy as np
        import os
        # Project the 3D model of the end-effectors
        # onto the 2D image plane using the provided camera parameters.
        rendered_image = image.copy()
        rendered_image = rendered_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        rendered_image = np.ascontiguousarray(rendered_image, dtype=np.uint8)
        # Example: Draw a simple circle at the projected position of the end-effectors
        
        # from scipy.spatial.transform import Rotation as R
        save_dir = "/home/yz12129/tmp/eef_render_debug"
        os.makedirs(save_dir, exist_ok=True)
        frame_idx = getattr(self, "_debug_frame_idx", 0)
        # cv2.imwrite(os.path.join(save_dir, f"frame_orig_{frame_idx:04d}.png"), rendered_image)
        
        # Project the 3D positions to 2D (this is a simplified example)
        projected_point1 = self.transform_world_to_pixel(eef1_pos[None, :], extrinsics, intrinsics)[0]
        projected_point2 = self.transform_world_to_pixel(eef2_pos[None, :], extrinsics, intrinsics)[0]

        x1, y1 = int(projected_point1[0]), int(projected_point1[1])
        x2, y2 = int(projected_point2[0]), int(projected_point2[1])
        # print("projected eef positions:", x1, y1, x2, y2, flush=True)
        cv2.circle(rendered_image, (x1, y1), 5, (0, 255, 0), -1)  # Draw green circle for eef1
        cv2.circle(rendered_image, (x2, y2), 5, (255, 0, 0), -1)  # Draw blue circle for eef2
        # render eef orientation
        # Draw orientation axes
        # Define axis length in meters
        axis_length = 0.05
        # Define axis endpoints in eef frame
        axes = np.eye(3) * axis_length  # (3, 3)
        # Eef1
        axes1_points = eef1_pos[None, :] + (eef1_rot @ axes.T).T  # (3, 3)
        # Project each axis endpoint to 2D
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X=red, Y=green, Z=blue (BGR)
            proj_axis1 = self.transform_world_to_pixel(axes1_points[i:i+1, :], extrinsics, intrinsics)[0]
            x1a, y1a = int(proj_axis1[0]), int(proj_axis1[1])
            cv2.line(rendered_image, (x1, y1), (x1a, y1a), color, 2)
        # Eef2
        axes2_points = eef2_pos[None, :] + (eef2_rot @ axes.T).T  # (3, 3)
        # Project each axis endpoint to 2D
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X=red, Y=green, Z=blue (BGR)
            proj_axis2 = self.transform_world_to_pixel(axes2_points[i:i+1, :], extrinsics, intrinsics)[0]
            x2a, y2a = int(proj_axis2[0]), int(proj_axis2[1])
            cv2.line(rendered_image, (x2, y2), (x2a, y2a), color, 2)
        # cv2.imwrite(os.path.join(save_dir, f"frame_axes_{frame_idx:04d}.png"), rendered_image)
        self._debug_frame_idx = frame_idx + 1
        return rendered_image

    def render_gt_bimanual_eef_on_image(self, rendered_image, gt_eef1_pos, gt_eef1_rot, gt_eef2_pos, gt_eef2_rot, intrinsics, extrinsics):
        """
        Render the ground truth end-effectors on a single image frame.
        :param image: Single image frame (H, W, 3)
        :param gt_eef1_pos: Ground truth End-effector 1 position (3,)
        :param gt_eef1_rot: Ground truth End-effector 1 rotation (3, 3) as rotation matrix
        :param gt_eef2_pos: Ground truth End-effector 2 position (3,)
        :param gt_eef2_rot: Ground truth End-effector 2 rotation (3, 3) as rotation matrix
        :param intrinsics: Camera intrinsics (3, 3)
        :param extrinsics: Camera extrinsics (4, 4)
        :return: Image with rendered ground truth end-effectors
        """
        import cv2
        import numpy as np
        import os
        # Project the 3D model of the end-effectors
        # onto the 2D image plane using the provided camera parameters.
        # rendered_image = image.copy()
        # rendered_image = rendered_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        # rendered_image = np.ascontiguousarray(rendered_image, dtype=np.uint8)
        # Example: Draw a simple circle at the projected position of the end-effectors
        
        # from scipy.spatial.transform import Rotation as R
        # save_dir = "/home/yz12129/tmp/eef_render_debug"
        # os.makedirs(save_dir, exist_ok=True)
        # frame_idx = getattr(self, "_debug_frame_idx", 0)
        # cv2.imwrite(os.path.join(save_dir, f"frame_orig_{frame_idx:04d}.png"), rendered_image)
        
        # Project the 3D positions to 2D (this is a simplified example)
        projected_point1 = self.transform_world_to_pixel(gt_eef1_pos[None, :], extrinsics, intrinsics)[0]
        projected_point2 = self.transform_world_to_pixel(gt_eef2_pos[None, :], extrinsics, intrinsics)[0]

        x1, y1 = int(projected_point1[0]), int(projected_point1[1])
        x2, y2 = int(projected_point2[0]), int(projected_point2[1])
        # print("projected gt eef positions:", x1, y1, x2, y2, flush=True)
        cv2.circle(rendered_image, (x1, y1), 5, (0, 255, 255), -1)
        cv2.circle(rendered_image, (x2, y2), 5, (255, 255, 0), -1)
        # render eef orientation
        # Draw orientation axes
        # Define axis length in meters
        axis_length = 0.05
        # Define axis endpoints in eef frame
        axes = np.eye(3) * axis_length  # (3, 3)
        # Eef1
        axes1_points = gt_eef1_pos[None, :] + (gt_eef1_rot @ axes.T).T  # (3, 3)
        # Project each axis endpoint to 2D
        for i, color in enumerate([(55, 55, 200), (55, 200, 55), (200, 55, 55)]):  # X=red, Y=green, Z=blue (BGR)
            proj_axis1 = self.transform_world_to_pixel(axes1_points[i:i+1, :], extrinsics, intrinsics)[0]
            x1a, y1a = int(proj_axis1[0]), int(proj_axis1[1])
            cv2.line(rendered_image, (x1, y1), (x1a, y1a), color, 2)
        # Eef2
        axes2_points = gt_eef2_pos[None, :] + (gt_eef2_rot @ axes.T).T  # (3, 3)
        # Project each axis endpoint to 2D
        for i, color in enumerate([(90, 90, 255), (90, 255, 90), (255, 90, 90)]):  # X=red, Y=green, Z=blue (BGR)
            proj_axis2 = self.transform_world_to_pixel(axes2_points[i:i+1, :], extrinsics, intrinsics)[0]
            x2a, y2a = int(proj_axis2[0]), int(proj_axis2[1])
            cv2.line(rendered_image, (x2, y2), (x2a, y2a), color, 2)
        # cv2.imwrite(os.path.join(save_dir, f"frame_axes_{frame_idx:04d}.png"), rendered_image)
        # self._debug_frame_idx = frame_idx + 1
        return rendered_image


    def render_all(self):
        """
        Render the end-effector on all image frames.
        :return: List of images with rendered end-effector
        """
        rendered_images = []
        for i in range(len(self.image_frames)):
            rendered_image = self.render_eef_on_image(
                self.image_frames[i],
                self.eef_pos[i],
                self.eef_rot[i],
                self.intrinsics[i],
                self.extrinsics[i]
            )
            rendered_images.append(rendered_image)
        return rendered_images
    
    def render_all_bimanual(self):
        """
        Render the end-effectors on all image frames.
        :return: List of images with rendered end-effectors
        """
        rendered_images = []
        for i in range(len(self.image_frames)):
            rendered_image = self.render_bimanual_eef_on_image(
                self.image_frames[i],
                self.eef_pos[i],
                self.eef_rot[i],
                self.eef2_pos[i],
                self.eef2_rot[i],
                self.intrinsics[i],
                self.extrinsics[i]
            )
            rendered_images.append(rendered_image)
        return rendered_images
    
    def render_all_gt_bimanual(self):
        """
        Render the ground truth end-effectors on all image frames.
        :return: List of images with rendered ground truth end-effectors
        """
        rendered_images = []
        for i in range(len(self.image_frames)):
            rendered_image = self.render_bimanual_eef_on_image(
                self.image_frames[i],
                self.eef_pos[i],
                self.eef_rot[i],
                self.eef2_pos[i],
                self.eef2_rot[i],
                self.intrinsics[i],
                self.extrinsics[i]
            )
            rendered_image = self.render_gt_bimanual_eef_on_image(
                rendered_image,
                self.gt_eef_pos[i],
                self.gt_eef_rot[i],
                self.gt_eef2_pos[i],
                self.gt_eef2_rot[i],
                self.intrinsics[i],
                self.extrinsics[i]
            )
            rendered_images.append(rendered_image)
        return rendered_images

    def save_rendered_video(self, rendered_images, output_path, fps=8):
        """
        Save the rendered images as a video.
        :param rendered_images: List of rendered images
        :param output_path: Path to save the video
        :param fps: Frames per second
        """
        import cv2
        height, width, _ = rendered_images[0].shape
        print("video length:", len(rendered_images), flush=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for img in rendered_images:
            video_writer.write(img)

        video_writer.release()
        print(f"Video saved to {output_path}", flush=True)

    def run(self, output_path):
        """
        Render end-effector on images, and save as video. Call this after consume_data.
        :param image_frames: List of image frames (T, H, W, 3)
        :param eef_pos: End-effector positions (T, 3)
        :param eef_rot: End-effector rotations (T, 4) as quaternions
        :param extrinsics: Camera extrinsics (T, 4, 4)
        :param intrinsics: Camera intrinsics (T, 3, 3)
        :param output_path: Path to save the video
        """
        print("Rendering and saving video to", output_path, flush=True)
        assert self.image_frames is not None, "Image frames not provided"
        assert self.eef_pos is not None, "End-effector positions not provided"
        assert self.eef_rot is not None, "End-effector rotations not provided"
        assert self.intrinsics is not None, "Camera intrinsics not provided"
        assert self.extrinsics is not None, "Camera extrinsics not provided"
        print("images shape:", self.image_frames.shape, flush=True)
        print("eef_pos shape:", self.eef_pos.shape, flush=True)
        print("eef_rot shape:", self.eef_rot.shape, flush=True)
        print("intrinsics shape:", self.intrinsics.shape, flush=True)
        print("extrinsics shape:", self.extrinsics.shape, flush=True)
        # rendered_images = self.render_all()
        rendered_images = self.render_all_gt_bimanual()
        self.save_rendered_video(rendered_images, output_path)

    def clear_data(self):
        """
        Clear all stored data.
        """
        self.image_frames = None
        self.eef_pos = None
        self.eef_rot = None
        self.extrinsics = None
        self.intrinsics = None
        self._debug_frame_idx = 0
        self.eef2_pos = None
        self.eef2_rot = None
        self.gt_eef_pos = None
        self.gt_eef_rot = None
        self.gt_eef2_pos = None
        self.gt_eef2_rot = None