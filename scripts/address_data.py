import os
import json
import pickle
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import open3d as o3d
class DataPreprocessor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.config = self._load_config()
        
    def _load_config(self):
        with open(self.source_dir / 'config.json', 'r') as f:
            return json.load(f)
    
    def process_episode(self, ep_num: int) -> dict:
        """处理单个episode数据"""
        # 读取pickle数据
        with open(self.source_dir / f'episode_{ep_num}.pkl', 'rb') as f:
            ep_data = pickle.load(f)
            
        if ep_data.get('score', 0) == 0:
            return None
            
        # 打印score
        print(f"Processing episode {ep_num}, score: {ep_data['score']}")
        # 读取图像和深度数据
        processed_data = {
            'image': [], 'depth': [], 'point_cloud': [],
            'agent_pos': ep_data['observation.state'],
            'action': ep_data['observation.last_action'] / 1000.0  # 归一化action
        }
        
        # 读取并处理图像和深度数据 
        # past
        # cap1_rgb = cv2.VideoCapture(str(self.source_dir / f'episode_{ep_num}.observation.image.camera2.mp4'))
        # cap1_depth = cv2.VideoCapture(str(self.source_dir / f'episode_{ep_num}.observation.depth.camera2.mp4'))
        depth_frames = self._load_depth_video(str(self.source_dir / f'episode_{ep_num}.observation.depth.camera2.mp4'))
        rgb_frames = self._load_rgb_video(str(self.source_dir / f'episode_{ep_num}.observation.image.camera2.mp4'))
        # type of depth_frames,rgb_frames
        # print(type(depth_frames)) <class 'list'>
        # print(type(rgb_frames)) <class 'list'>
        # print(type(depth_frames[0]), type(rgb_frames[0])) <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(len(depth_frames), len(rgb_frames)) 334 334
        # print(depth_frames[0].shape, rgb_frames[0].shape) (480, 640) (480, 640, 3)
        # exit()
        processed_data['image'] = rgb_frames
        processed_data['depth'] = depth_frames
        
        width, height = 640, 480
        camera_intrinsic = np.array(self.config['camera_intrinsic.camera2'])
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        for i in tqdm(range(len(depth_frames)), desc="Processing frames"):
            new_pcd = self._get_pcd_from_rgbd(rgb_frames[i], depth_frames[i], pinhole_camera_intrinsic)
            xyz = np.asarray(new_pcd.points)
            rgb = np.asarray(new_pcd.colors)
            # print(type(xyz), type(rgb)) open3d.cuda.pybind.utility.Vector3dVector
            # convert open3d.cuda.pybind.utility.Vector3dVector to numpy.ndarray
            # print(np.asarray(xyz).shape, np.asarray(rgb).shape) (288857, 3) (288857, 3)
            points = np.concatenate([xyz, rgb], axis=-1)
            processed_data['point_cloud'].append(points)
        # for t in trange(100000):
        #     i = t % len(self.rgb_frames)
        #     new_pcd = self.get_pcd_from_rgbd(self.rgb_frames[i], self.depth_frames[i], self.pinhole_camera_intrinsic)
        #     pcd.points = new_pcd.points
        #     pcd.colors = new_pcd.colors
        #     vis.update_geometry(pcd)  # 更新点云
        #     vis.poll_events()         # 处理事件
        #     vis.update_renderer()     # 更新渲染器
        #     time.sleep(interval)      # 控制帧间隔

        

        # while True:
        #     ret1_rgb, frame1_rgb = cap1_rgb.read()
        #     ret1_depth, frame1_depth = cap1_depth.read()
            
        #     if not (ret1_rgb and ret1_depth):
        #         break
                
        #     processed_data['image'].append(frame1_rgb)
        #     processed_data['depth'].append(frame1_depth[..., 0])  # 只取一个通道
        #     # 这里需要生成point cloud，使用camera_intrinsic
        #     point_cloud = self._get_pcd_from_rgbd(frame1_rgb, frame1_depth, self.config['camera_intrinsic.camera2'])
        #     processed_data['point_cloud'].append(point_cloud)

            
        return processed_data
    
    # mp4的数值范围是0-255，原始深度范围是0-1500mm
    def _load_depth_video(self, input_path, max_depth=1500):
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_path}")
    
        depth_frames = []  # 存储深度帧序列

        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break
        
           # 检查是否是灰度图
            if len(frame.shape) == 3 and frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        
            # 将 0-255 的值映射回原始深度值
            frame_depth = (frame.astype(np.float32) * max_depth / 255).astype(np.float32)
            depth_frames.append(frame_depth)
    
        cap.release()  # 释放资源
        return depth_frames

    def _load_rgb_video(self, input_path):
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_path}")
    
        frames = []  # 用于存储 RGB 帧

        while True:
            ret, frame = cap.read()  # 逐帧读取视频
            if not ret:
                break  # 当没有更多帧时退出循环
        
            # OpenCV 默认读取 BGR 图像，将其转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
        cap.release()  # 释放视频资源
        return frames
    
    def _get_pcd_from_rgbd(self, color_image, depth_image, pinhole_camera_intrinsic):
        #print(depth_image.shape, color_image.shape)
        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        return pcd
    
    # def _generate_point_cloud(self, rgb, depth, intrinsic):
    #     """生成彩色点云"""
    #     height, width = depth.shape[:2]
    #     fx, fy = intrinsic[0][0], intrinsic[1][1]
    #     cx, cy = intrinsic[0][2], intrinsic[1][2]
        
    #     # 生成像素坐标
    #     x = np.arange(width)
    #     y = np.arange(height)
    #     xx, yy = np.meshgrid(x, y)
        
    #     # 计算3D点
    #     z = depth[..., 0].astype(float)
    #     x = (xx - cx) * z / fx
    #     y = (yy - cy) * z / fy
        
    #     # 组合xyz和rgb
    #     xyz = np.stack([x, y, z], axis=-1)
    #     points = np.concatenate([xyz, rgb], axis=-1)
        
    #     # 过滤无效点
    #     valid_mask = z > 0
    #     points = points[valid_mask]
        
    #     return points
    
    def process_all(self):
        """处理所有数据"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing data from {self.source_dir}")
        print(f"Output directory: {self.output_dir}\n")
        
        # 复制config文件
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f)
        print("Config file copied")
                
        # 处理每个episode 
        episodes = sorted([int(f.stem.split('_')[1]) for f in self.source_dir.glob('episode_*.pkl')])
        print(f"Found {len(episodes)} episodes")
        
        successful = 0
        for ep_num in tqdm(episodes, desc="Processing episodes"):
            data = self.process_episode(ep_num)
            if data is not None:  # 只保存成功的轨迹
                episode_dir = self.output_dir / str(ep_num)
                episode_dir.mkdir(parents=True, exist_ok=True)
                
                with open(episode_dir / 'data.pkl', 'wb') as f:
                    pickle.dump(data, f)
                successful += 1
        
        print(f"\nProcessing complete:")
        print(f"Total episodes: {len(episodes)}")
        print(f"Successful episodes: {successful}")
        print(f"Failed episodes: {len(episodes) - successful}")

if __name__ == "__main__":
    source_dir="/raid/sunyuanxu/teleoperation_data/PolePickPlace_RM65_Inspire_left_20241106"
    output_dir="/raid/sunyuanxu/teleoperation_data/PolePickPlace_RM65_Inspire_left_20241106_processed_dataset_1118_v1_test"
    preprocessor = DataPreprocessor(
        source_dir=source_dir,
        output_dir=output_dir
    )
    preprocessor.process_all()
    # with open("/home/sunyuanxu/code/teleoperation_data/PolePickPlace_RM65_Inspire_left_20241106" + '/episode_1.pkl', 'rb') as f:
    #     ep_data = pickle.load(f)
    #     # 存储 ep_data 的每个的key和value的形状
    #     with open("/home/sunyuanxu/code/teleoperation_data/PolePickPlace_RM65_Inspire_left_20241106" + '/episode_1_shape.txt', 'w') as f:
    #         for key, value in ep_data.items():
    #             if hasattr(value, 'shape'):
    #                 f.write(f"{key}(shape:{value.shape}): {value}\n")
    #             else:
    #                 f.write(f"{key}: {value}\n")

    # 查看存储之后的某一个的结果,只查看点云信息,并一个txt到当前路径
    with open(output_dir + '/1/data.pkl', 'rb') as f:
        data = pickle.load(f)
        with open('data_shape.txt', 'w') as f:
            # point_cloud
            for point_cloud in data['point_cloud']:
                f.write(f"point_cloud(shape:{point_cloud.shape}): {point_cloud}\n")
                break
            
# import numpy as np
# import open3d as o3d
# import time
# import cv2

# class Test:
#     def __init__(self, rgb_path, depth_path, camera_intrinsic):
#         self.depth_frames = self.load_depth_video(depth_path)
#         self.rgb_frames = self.load_rgb_video(rgb_path)
#         #print(self.rgb_frames)
#         self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
#         width, height = 640, 480
#         fx = camera_intrinsic[0, 0]
#         fy = camera_intrinsic[1, 1]
#         cx = camera_intrinsic[0, 2]
#         cy = camera_intrinsic[1, 2]
#         self.pinhole_camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
#         print(self.pinhole_camera_intrinsic.intrinsic_matrix)

#         #pcd = self.get_pcd_from_rgbd(self.rgb_frames[0], self.depth_frames[0], self.pinhole_camera_intrinsic)
#         self.visualize_point_cloud_sequence()


#     def load_rgb_video(self, input_path):
#         # 打开视频文件
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             raise ValueError(f"无法打开视频文件：{input_path}")
    
#         frames = []  # 用于存储 RGB 帧

#         while True:
#             ret, frame = cap.read()  # 逐帧读取视频
#             if not ret:
#                 break  # 当没有更多帧时退出循环
        
#             # OpenCV 默认读取 BGR 图像，将其转换为 RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
    
#         cap.release()  # 释放视频资源
#         return frames

#     # mp4的数值范围是0-255，原始深度范围是0-1500mm
#     def load_depth_video(self, input_path, max_depth=1500):
#         # 打开视频文件
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             raise ValueError(f"无法打开视频文件：{input_path}")
    
#         depth_frames = []  # 存储深度帧序列

#         while True:
#             ret, frame = cap.read()  # 读取一帧
#             if not ret:
#                 break
        
#            # 检查是否是灰度图
#             if len(frame.shape) == 3 and frame.shape[-1] == 3:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        
#             # 将 0-255 的值映射回原始深度值
#             frame_depth = (frame.astype(np.float32) * max_depth / 255).astype(np.float32)
#             depth_frames.append(frame_depth)
    
#         cap.release()  # 释放资源
#         return depth_frames


#     def get_pcd_from_rgbd(self, color_image, depth_image, pinhole_camera_intrinsic):
#         #print(depth_image.shape, color_image.shape)
#         depth = o3d.geometry.Image(depth_image)
#         color = o3d.geometry.Image(color_image)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
#         return pcd

    # def visualize_point_cloud_sequence(self, interval=0.1):
    #     # 创建一个可视化窗口
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()

    #     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    #     vis.add_geometry(coordinate_frame)
    
    #     pcd = o3d.geometry.PointCloud()  # 创建一个空的点云
    #     vis.add_geometry(pcd)

    #     # 设置观察方向
    #     view_control = vis.get_view_control()
    #     view_control.set_lookat([-0.1, -0.1, 0.3])  # 观察点
    #     view_control.set_front([0, 0, -1])  # 前视方向Z
    #     view_control.set_up([0, -1, 0])  # 上方向设置-Y
    #     view_control.set_zoom(1)  # 设置缩放比例以适应场景
    
    #     from tqdm import trange
    #     for t in trange(100000):
    #         i = t % len(self.rgb_frames)
    #         new_pcd = self.get_pcd_from_rgbd(self.rgb_frames[i], self.depth_frames[i], self.pinhole_camera_intrinsic)
    #         pcd.points = new_pcd.points
    #         pcd.colors = new_pcd.colors
    #         vis.update_geometry(pcd)  # 更新点云
    #         vis.poll_events()         # 处理事件
    #         vis.update_renderer()     # 更新渲染器
    #         time.sleep(interval)      # 控制帧间隔
    
    #     vis.destroy_window()  # 销毁窗口
    

# if __name__=='__main__':
#     import json
#     with open('PolePickPlace_RM65_Inspire_left_20241106/config.json', 'r') as f:
#         config = json.load(f)
#     camera_intrinsic = np.array(config['camera_intrinsic.camera2'])
#     episode = 10
#     test = Test('PolePickPlace_RM65_Inspire_left_20241106/episode_{}.observation.image.camera2.mp4'.format(episode), 
#                 'PolePickPlace_RM65_Inspire_left_20241106/episode_{}.observation.depth.camera2.mp4'.format(episode),
#                 camera_intrinsic)