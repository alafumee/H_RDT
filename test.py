import os
import random
import h5py
import numpy as np
import cv2
import json
import pandas as pd
import time
import threading
import glob
from typing import List, Dict
import sys
import warnings
import traceback

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*multichannel.*"
)

# 导入imgaug相关库
import imgaug as ia
import imgaug.augmenters as iaa

if not hasattr(np, 'bool'):
    np.bool = bool

class RobotwinAgilexDataset:
    """
    Dataset for loading RobotWin Agilex robot data with joint actions.
    支持多任务均衡采样和定期更新数据集。
    """
    def __init__(
        self, 
        robot_root_dir="/share/hongzhe/RoboTwin/data",
        task_list=None,  # 如果为None，则使用根目录下的所有任务文件夹
        config=None,
        stat_path='/share/hongzhe/rdtv_0423/datasets/robotwin/stat/cvpr.json',
        lang_path='/share/hongzhe/rdtv_0423/datasets/robotwin_round2/robotwin_instructions_cvpr2025_workshop.csv',
        upsample_rate=3,
        val=False,
        image_aug=False,
        image_corrupt_severity=5,
        use_precomp_lang_embed=False,
        embed_dir='/share/hongzhe/rdtv_0423/datasets/robotwin_round2/embeddings',
        update_interval=3600  # 更新间隔，默认3600秒（1小时）
    ):
        """
        初始化RobotWin数据集，支持多任务均衡采样。
        
        Args:
            robot_root_dir: RobotWin数据根目录，包含多个任务文件夹
            task_list: 任务名称列表，如果为None则使用根目录下的所有任务文件夹
            config: 配置字典
            stat_path: 用于归一化的统计文件路径
            lang_path: 语言指令文件路径
            upsample_rate: 时序数据的上采样率
            val: 是否为验证集
            image_aug: 是否应用图像增强
            image_corrupt_severity: 图像损坏的严重程度
            update_interval: 更新数据集的时间间隔（秒）
        """
        self.DATASET_NAME = "robotwin_agilex"
        self.robot_root_dir = robot_root_dir
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_aug = image_aug
        self.image_corrupt_severity = image_corrupt_severity
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.embed_dir = embed_dir
        self.update_interval = update_interval
        
        # 设置基本参数
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = config['common']['action_dim']
        self.img_history_size = config['common']['img_history_size']
        
        # 加载统计数据用于归一化
        with open(stat_path, 'r') as file:
            stat = json.load(file)
        self.action_min = np.array(stat['robotwin_agilex']['min'])
        self.action_max = np.array(stat['robotwin_agilex']['max'])
        
        # 加载语言指令数据（如果不使用预计算嵌入）
        if not self.use_precomp_lang_embed:
            self.lang_df = pd.read_csv(lang_path)
            self.instructions_map = dict(zip(self.lang_df['name'], self.lang_df['instruction']))
        
        # 设置相机参数
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 3:
            self.cameras = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
            self.segs = ["seg_high", "seg_right_wrist", "seg_left_wrist"]
        elif self.num_cameras == 4:
            self.cameras = ["cam_high", "cam_front", "cam_right_wrist", "cam_left_wrist"]
            self.segs = ["seg_high", "seg_front", "seg_right_wrist", "seg_left_wrist"]
        else:
            raise ValueError(f"不支持的相机数量 num_cameras={self.num_cameras}，应为3或4。")

        self.camera_mapping = {
            "cam_front": "front_camera",
            "cam_high": "head_camera",
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera"
        }
        
        # 初始化任务列表
        if task_list is None:
            # 获取根目录下的所有文件夹作为任务列表，排除seed文件夹
            self.task_list = [d for d in os.listdir(self.robot_root_dir) 
                              if os.path.isdir(os.path.join(self.robot_root_dir, d)) and d != "seed"]
        else:
            self.task_list = task_list
        
        print(f"任务列表: {self.task_list}")
        
        # 初始化数据结构
        self.task_to_episodes = {}  # 每个任务对应的数据集
        self.task_weights = {}      # 每个任务的采样权重
        self.total_episodes = 0     # 所有任务的总样本数
        self.last_update_time = 0   # 上次更新时间
        
        # 初始化数据集并启动更新线程
        self._update_dataset()
        
        # 启动定期更新线程
        if update_interval > 0:
            self.update_thread = threading.Thread(target=self._periodic_update, daemon=True)
            self.update_thread.start()
    
    def _scan_task_folder(self, task_name):
        """扫描指定任务文件夹，获取所有hdf5文件路径"""
        task_dir = os.path.join(self.robot_root_dir, task_name)
        if not os.path.exists(task_dir):
            print(f"警告: 任务文件夹 {task_dir} 不存在")
            return []
        
        # 搜索两层目录结构下的所有hdf5文件
        hdf5_files = []
        
        # 获取任务目录下的所有子目录
        subdirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(task_dir, subdir)
            # 在子目录中查找所有hdf5文件
            for f in os.listdir(subdir_path):
                if f.endswith(".hdf5"):
                    hdf5_path = os.path.join(subdir_path, f)
                    hdf5_files.append(hdf5_path)
        
        print(f"任务 {task_name}: 找到 {len(hdf5_files)} 个hdf5文件")
        return hdf5_files
    
    def _update_dataset(self):
        """更新数据集，重新扫描所有任务文件夹并更新采样权重"""
        print("正在更新数据集...")
        
        # 更新每个任务的数据集
        all_task_count = 0
        self.task_to_episodes = {}
        
        for task_name in self.task_list:
            # 扫描任务文件夹获取所有hdf5文件
            episodes = self._scan_task_folder(task_name)
            # 随机打乱文件顺序
            random.shuffle(episodes)
            # 更新任务对应的数据集
            self.task_to_episodes[task_name] = episodes
            # 统计总样本数
            all_task_count += len(episodes)
        
        # 如果没有找到任何数据，报错
        if all_task_count == 0:
            raise ValueError("错误: 未找到任何hdf5文件，请检查数据路径")
        
        # 计算每个任务的采样权重 - 所有任务权重相同，确保1:1:1:1:1:1采样
        task_count = len(self.task_list)
        for task_name in self.task_list:
            self.task_weights[task_name] = 1.0 / task_count
        
        self.total_episodes = all_task_count
        self.last_update_time = time.time()
        
        print(f"数据集更新完成。总共 {all_task_count} 个样本，分布在 {task_count} 个任务中")
        print(f"任务权重: {self.task_weights}")
    
    def _periodic_update(self):
        """周期性更新数据集的线程函数"""
        while True:
            # 等待指定的更新间隔
            time.sleep(self.update_interval)
            try:
                # 更新数据集
                self._update_dataset()
            except Exception as e:
                print(f"数据集更新过程中出错: {e}")
                traceback.print_exc()
    
    def __len__(self):
        """返回数据集的近似长度"""
        return self.total_episodes * 200  # 每个hdf5文件假设有200个样本
    
    def get_dataset_name(self):
        """返回数据集名称"""
        return self.DATASET_NAME

    def parse_img_data(self, dataset, idx):
        """
        处理单个相机的图像数据
        
        Args:
            dataset: 单个相机的HDF5数据集
            idx: 当前帧索引
            
        Returns:
            处理后的图像序列，形状为 [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # 使用原始分辨率 240x320
        frames = np.zeros((num_frames, 240, 320, 3), dtype=np.uint8)
        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < len(dataset):
                    img_data = dataset[frame_idx]

                    decoded_img = self.decode_image_with_pillow(img_data)
                    if decoded_img is None:
                        raise Exception(f"[DEBUG] decode error")

                    if decoded_img is not None:
                        frames[i] = decoded_img

        except Exception as e:
            print(f"[DEBUG] decode_image_with_pillow error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames])
        
        return frames

    def decode_image_with_pillow(self, img_data):
        """
        使用OpenCV解码图像数据，保持RGB格式
        
        Args:
            img_data (bytes): 二进制图像数据
        
        Returns:
            np.ndarray: 解码后的图像数组，shape=(240, 320, 3)，RGB格式
        """
        try:
            # 使用OpenCV直接解码
            nparr = np.frombuffer(img_data, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 确保解码成功
            if bgr_img is None:
                raise Exception("OpenCV解码失败")
                
            # OpenCV默认使用BGR格式，需要转换为RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸到 240x320 (如果需要)
            if rgb_img.shape[:2] != (240, 320):
                rgb_img = cv2.resize(rgb_img, (320, 240))
                
            return rgb_img
            
        except Exception as e:
            print(f"[DEBUG] 图像解码失败: {e}")
            return None

    def parse_seg_data(self, dataset, idx):
        """
        处理分割掩码数据。解码JPEG字节流为单通道掩码
        
        Args:
            dataset: HDF5数据集，包含分割掩码数据的字节流
            idx: 当前帧索引
            
        Returns:
            处理后的分割掩码序列 [history_size, 240, 320]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        seg_frames = np.zeros((num_frames, 240, 320), dtype=np.uint8)
        
        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < len(dataset):
                    # 分割掩码是JPEG编码的字节流
                    seg_data = dataset[frame_idx]
                    try:
                        # 使用OpenCV直接解码字节流
                        nparr = np.frombuffer(seg_data, np.uint8)
                        seg_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                        
                        if seg_img is not None:
                            # 确保尺寸是240x320
                            if seg_img.shape != (240, 320):
                                seg_img = cv2.resize(seg_img, (320, 240))
                            
                            seg_frames[i] = seg_img
                    except Exception as e:
                        print(f"[DEBUG] Failed to decode seg image: {e}")
                        # 解码失败则使用零掩码
        except Exception as e:
            print(f"[DEBUG] parse_seg_data error: {e}")

        # 如果实际帧数不足，使用第一帧填充
        if num_frames < self.img_history_size:
            pad = np.repeat(seg_frames[:1], self.img_history_size - num_frames, axis=0)
            seg_frames = np.concatenate([pad, seg_frames])
        
        return seg_frames[:self.img_history_size]
            
    def get_language_description(self, task_type):
        """
        根据任务类型获取语言指令或预计算的嵌入文件路径
        
        Args:
            task_type (str): 任务类型名称
        
        Returns:
            如果use_precomp_lang_embed为True，返回预计算嵌入文件路径；
            否则返回任务对应的语言指令文本
        """
        if self.use_precomp_lang_embed:
            # 返回预计算的语言嵌入文件路径
            embed_path = os.path.join(self.embed_dir, f"{task_type}.pt")
            return embed_path
        else:
            # 从instructions_map中获取指令文本
            return self.instructions_map.get(task_type, f"Execute the {task_type} task")

    def extract_episode_item(self, hdf5_file):
        """
        从HDF5文件中提取单个样本的数据
        
        Args:
            hdf5_file: HDF5文件路径
            
        Returns:
            包含提取数据的字典，若提取失败则返回None
        """
        try:
            with h5py.File(hdf5_file, 'r', swmr=True, libver='latest') as f:
                # 新的HDF5结构使用joint_action替代qpos
                # 获取各个部分并拼接
                try:
                    left_arm = f["joint_action/left_arm"][:]
                    left_gripper = f["joint_action/left_gripper"][:]
                    right_arm = f["joint_action/right_arm"][:]
                    right_gripper = f["joint_action/right_gripper"][:]
                    
                    # 根据维度情况处理数据
                    # 如果左臂是2D而左夹爪是1D，需要扩展夹爪维度
                    if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                        left_gripper = left_gripper.reshape(-1, 1)
                    
                    # 如果右臂是2D而右夹爪是1D，需要扩展夹爪维度
                    if len(right_arm.shape) == 2 and len(right_gripper.shape) == 1:
                        right_gripper = right_gripper.reshape(-1, 1)
                    
                    # 拼接各个部分为一个完整的动作向量
                    action_data = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                    
                except Exception as e:
                    print(f"加载joint action数据时出错: {e}")
                    return None
                
                # 调整数据索引方法
                max_index = len(action_data) - 2
                index = random.randint(0, max_index)
                
                # 当前状态 (使用joint action数据)
                action_current = action_data[index]
                
                # 未来动作序列
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_chunk = action_data[index+1:action_end+1:self.upsample_rate]
                
                # 如果动作序列不足chunk_size，重复最后一帧填充
                if action_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(action_chunk[-1:], self.chunk_size - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, last_part], axis=0)
                
                # 获取相机数据
                try:
                    current_images = []
                    current_segs = []
                    
                    # 更新相机路径以匹配新的HDF5结构
                    camera_paths = {
                        "cam_front": "observation/front_camera/rgb",
                        "cam_high": "observation/head_camera/rgb",
                        "cam_left_wrist": "observation/left_camera/rgb",
                        "cam_right_wrist": "observation/right_camera/rgb"
                    }
                    
                    seg_paths = {
                        "seg_front": "observation/front_camera/actor_segmentation",
                        "seg_high": "observation/head_camera/actor_segmentation",
                        "seg_left_wrist": "observation/left_camera/actor_segmentation",
                        "seg_right_wrist": "observation/right_camera/actor_segmentation"
                    }
                    
                    # 根据num_cameras选择相机
                    for cam_idx, cam_name in enumerate(self.cameras):
                        # 获取相机图像
                        cam_path = camera_paths.get(cam_name)
                        if cam_path and cam_path in f:
                            camera_data = f[cam_path]
                            img_frames = self.parse_img_data(camera_data, index)
                            
                            # 获取对应的分割掩码
                            seg_name = self.segs[cam_idx]
                            seg_path = seg_paths.get(seg_name)
                            if seg_path and seg_path in f:
                                seg_frames = self.parse_seg_data(f[seg_path], index)
                            else:
                                print(f"警告: 分割掩码 {seg_name} 在 {hdf5_file} 中未找到")
                                return None

                            current_images.append(img_frames)
                            current_segs.append(seg_frames)
                        else:
                            print(f"警告: 相机 {cam_name} 在 {hdf5_file} 中未找到")
                            return None
                    
                    # 确保相机数量正确
                    if len(current_images) != self.num_cameras:
                        print(f"错误: 期望 {self.num_cameras} 个相机, 但得到 {len(current_images)}")
                        return None
                    
                    # 如果启用图像增强，应用增强
                    if self.image_aug and not self.val:
                        aug_current_images = []
                        for i in range(len(current_images)):
                            cam_images = []
                            for h in range(self.img_history_size):
                                img = current_images[i][h]  # 获取RGB图像
                                seg_mask = current_segs[i][h]   # 获取分割掩码
                                
                                # 使用分割掩码对背景进行增强
                                processed_img = self.process_background_with_imgaug(img, seg_mask)
                                cam_images.append(processed_img)
                            
                            aug_current_images.append(np.array(cam_images))
                        
                        # 使用增强后的图像
                        final_images = np.array(aug_current_images)
                    else:
                        # 不使用增强，直接使用原始图像
                        final_images = np.array(current_images)
                    
                    # 创建每个相机的掩码
                    mask_length = self.img_history_size
                    current_images_mask = [
                        np.array([True]*mask_length, dtype=bool) for _ in range(self.num_cameras)
                    ]
                    
                except Exception as e:
                    print(f"访问 {hdf5_file} 中的相机数据时出错: {e}")
                    traceback.print_exc()
                    return None
                
                # 从文件路径提取任务名称
                # 目录结构: /share/hongzhe/RoboTwin/data/[任务名]/[配置目录]/xxx.hdf5
                # 提取任务名称为路径中倒数第二级目录
                path_parts = hdf5_file.split(os.sep)
                task_name = path_parts[-3]  # 倒数第三个部分是任务名
                
                # 获取语言描述
                task_desc = self.get_language_description(task_name)
                
                state_indicator = np.ones_like(action_current)
                action_norm = np.ones_like(action_chunk)
                
                # 创建结果字典
                result = {
                    "current_images": final_images,  # 当前帧图像（可能经过增强）
                    "current_images_mask": current_images_mask,  # 图像掩码
                    "current_segs": np.array(current_segs),  # 分割掩码
                    "actions": action_chunk,  # 动作序列
                    "states": np.expand_dims(action_current, axis=0),  # 状态
                    "state_indicator": state_indicator,
                    "action_norm": action_norm,
                    "instruction": task_desc,  # 语言指令
                    "dataset_name": self.DATASET_NAME,  # 数据集名称
                }
                
                # 如果做了图像增强，保存原始图像用于调试
                if self.image_aug:
                    result["original_images"] = np.array(current_images)
                
                return result

        except Exception as e:
            print(f"处理 {hdf5_file} 时出错: {e}")
            traceback.print_exc()
            return None

    def process_background_with_imgaug(self, image, segmentation_mask):
        """
        对图像背景应用增强，保持前景不变
        
        Args:
            image (np.ndarray): 输入RGB图像，形状为(H, W, 3)
            segmentation_mask (np.ndarray): 分割掩码，形状为(H, W)，非零值表示前景
            
        Returns:
            np.ndarray: 增强后的RGB图像，形状为(H, W, 3)，前景保持不变
        """
        # 创建前景掩码(二值)
        foreground_mask = segmentation_mask > 0
        
        # 构建imgaug增强序列
        # 1. 噪声增强器
        noise_augs = iaa.OneOf([
            # 基本噪声
            iaa.AdditiveGaussianNoise(scale=(10, 60)),          # 高斯噪声
            iaa.AdditivePoissonNoise(lam=(15, 30)),             # 泊松噪声
            iaa.AdditiveLaplaceNoise(scale=(10, 30)),           # 拉普拉斯噪声
            iaa.ImpulseNoise(p=(0.01, 0.1)),                    # 脉冲噪声
            iaa.SaltAndPepper(p=(0.02, 0.1)),                   # 椒盐噪声
            iaa.Salt(p=(0.02, 0.1)),                            # 盐噪声
            iaa.Pepper(p=(0.02, 0.1)),                          # 椒噪声
            
            # 粗噪声
            iaa.CoarseSaltAndPepper(p=(0.02, 0.1), size_percent=(0.01, 0.05)),  # 粗椒盐噪声
            iaa.CoarseSalt(p=(0.02, 0.1), size_percent=(0.01, 0.05)),           # 粗盐噪声
            iaa.CoarsePepper(p=(0.02, 0.1), size_percent=(0.01, 0.05)),         # 粗椒噪声
            iaa.CoarseDropout(p=(0.02, 0.15), size_percent=(0.02, 0.15)),       # 粗随机丢弃
        ])
        
        # 2. 模糊增强器
        blur_augs = iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.5, 3.0)),                  # 高斯模糊
            iaa.AverageBlur(k=(2, 7)),                           # 平均模糊
            iaa.MedianBlur(k=(3, 7)),                            # 中值模糊
            iaa.MotionBlur(k=(3, 7)),                            # 运动模糊
        ])
        
        # 3. 天气增强器
        weather_augs = iaa.OneOf([
            iaa.Clouds(),                                         # 云
            iaa.Fog(),                                            # 雾
            iaa.Snowflakes(density=(0.01, 0.05), density_uniformity=(0.3, 0.8)), # 雪花
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.1, 0.3)),     # 雨
        ])
        
        # 使用正确的概率设置方法
        aug = iaa.Sequential([
            # 每个增强器有不同的概率被应用
            iaa.Sometimes(0.9, noise_augs),     # 90%概率选择噪声类型
            iaa.Sometimes(0.05, blur_augs),     # 5%概率选择模糊类型
            iaa.Sometimes(0.05, weather_augs)   # 5%概率选择天气类型
        ], random_order=True)  # 随机顺序应用
        
        # 复制原始图像进行单独处理
        result = image.copy()
        
        # 只处理背景部分
        background = image.copy()
        if len(foreground_mask.shape) == 2 and len(background.shape) == 3:
            # 扩展掩码维度以匹配图像
            foreground_mask = np.expand_dims(foreground_mask, axis=-1)
            foreground_mask = np.repeat(foreground_mask, 3, axis=-1)
        
        # 将前景区域设为黑色
        background[foreground_mask] = 0
        
        # 应用imgaug增强器
        corrupted_background = aug.augment_image(background)
        
        # 合并原始前景与处理后的背景
        background_mask = ~foreground_mask
        result[background_mask] = corrupted_background[background_mask]
        
        return result

    def get_item(self, index=None):
        """
        获取数据项，如果index为None则随机选择一个
        
        Args:
            index: 可选，指定索引。如果为None，则随机选择
            
        Returns:
            处理后的数据字典，失败则返回None
        """
        if not self.task_to_episodes:
            # 如果数据集为空，更新数据集
            self._update_dataset()
        
        # 检查是否需要更新数据集（时间间隔超过update_interval）
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            # 在单独的线程中更新数据集，避免阻塞当前操作
            update_thread = threading.Thread(target=self._update_dataset)
            update_thread.daemon = True
            update_thread.start()
        
        # 基于任务权重随机选择一个任务
        task_name = random.choices(
            list(self.task_weights.keys()),
            weights=list(self.task_weights.values()),
            k=1
        )[0]
        
        # 从选中的任务中随机选择一个样本
        task_episodes = self.task_to_episodes.get(task_name, [])
        if not task_episodes:
            print(f"警告: 任务 {task_name} 没有可用的样本")
            # 从其他任务中选择
            alternative_tasks = [t for t in self.task_list if t != task_name and self.task_to_episodes.get(t, [])]
            if not alternative_tasks:
                print("错误: 没有任何可用的样本")
                return None
            task_name = random.choice(alternative_tasks)
            task_episodes = self.task_to_episodes.get(task_name, [])
        
        # 随机选择一个hdf5文件
        episode_file = random.choice(task_episodes)
        
        # 尝试提取样本数据
        for _ in range(3):  # 最多尝试3次
            item = self.extract_episode_item(episode_file)
            if item is not None:
                return item
            # 如果当前样本提取失败，随机选择另一个
            episode_file = random.choice(task_episodes)
        
        print(f"警告: 从任务 {task_name} 中提取样本失败，返回None")
        return None

if __name__ == "__main__":
    # 测试代码
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='测试RobotwinAgilexDataset')
    parser.add_argument('--config', type=str, default='/share/hongzhe/VLA/round2/dino_siglip/configs/rdtv.yaml', help='配置文件路径')
    parser.add_argument('--update_interval', type=int, default=3600, help='数据集更新间隔(秒)')
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建数据集实例
    print("创建数据集实例...")
    ds = RobotwinAgilexDataset(
        config=config, 
        val=False,
        update_interval=args.update_interval
    )
    
    # 测试加载数据
    print("\n测试加载数据...")
    success_count = 0
    test_times = 10
    
    for i in range(test_times):
        print(f"\n尝试 {i+1}/{test_times}...")
        item = ds.get_item()
        
        if item is not None:
            success_count += 1
            print("成功加载数据")
            print(f"指令: {item['instruction']}")
    
    print(f"\n结果: 成功加载 {success_count}/{test_times} 个样本") 