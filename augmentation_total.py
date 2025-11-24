import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os
from dataset import CATEGORY_LIST


# Time Warping
def time_warp(data: np.ndarray, warp_factor: float = 1.5) -> np.ndarray:
    
    n_timesteps = len(data)
    original_indices = np.arange(n_timesteps)

   
    random_warps = np.random.normal(loc=1.0, scale=warp_factor * 0.1, size=(5,))
    warp_steps = np.linspace(0, n_timesteps - 1, num=5)

    
    time_warp_func = CubicSpline(warp_steps, warp_steps * random_warps)
    warped_indices = time_warp_func(original_indices)

   
    warped_indices = np.clip(warped_indices, 0, n_timesteps - 1)

    
    warped_data = np.zeros_like(data)
    for i in range(3):  
        interpolator = CubicSpline(original_indices, data[:, i])
        warped_data[:, i] = interpolator(warped_indices)

    return warped_data


# Add Jitter 
def add_jitter(data: np.ndarray, sigma: float = 0.03) -> np.ndarray:
   
    
    data_std = np.std(data, axis=0)
    noise = np.random.normal(loc=0, scale=sigma * data_std, size=data.shape)

    jittered_data = data + noise
    return jittered_data


# Rotate 3D 
def rotate_3d(data: np.ndarray, angle_deg: float, axis: str = 'z') -> np.ndarray:
   
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    if axis.lower() == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ])
    elif axis.lower() == 'y':
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
    elif axis.lower() == 'z':
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

 
    rotated_data = data @ rotation_matrix.T
    return rotated_data


#  Scale Data 
def scale_data(data: np.ndarray, scale_factor: float = 1.2) -> np.ndarray:
   
   
    center = np.mean(data, axis=0)

    
    scaled_data = center + (data - center) * scale_factor

    return scaled_data


#  Time Shifting 
def time_shift(data: np.ndarray, shift_ratio: float = 0.1) -> np.ndarray:
    
    n_timesteps = len(data)
    shift_amount = int(n_timesteps * shift_ratio)

   
    shifted_data = np.roll(data, shift_amount, axis=0)

    return shifted_data


#  통합 증강 함수 
def augment_single_data(data: np.ndarray, method: str, **kwargs) -> np.ndarray:
   
    if method == 'time_warp':
        return time_warp(data, warp_factor=kwargs.get('warp_factor', 1.5))
    elif method == 'jitter':
        return add_jitter(data, sigma=kwargs.get('sigma', 0.03))
    elif method == 'rotate':
        return rotate_3d(data, angle_deg=kwargs.get('angle_deg', 15),
                        axis=kwargs.get('axis', 'z'))
    elif method == 'scale':
        return scale_data(data, scale_factor=kwargs.get('scale_factor', 1.2))
    elif method == 'shift':
        return time_shift(data, shift_ratio=kwargs.get('shift_ratio', 0.1))
    else:
        raise ValueError(f"Unknown augmentation method: {method}")


#DataFrame 증강 함수
def augment_dataframe(df: pd.DataFrame,
                     methods: list,
                     augment_per_method: int = 1) -> pd.DataFrame:
    
    all_data = [df]  

   
    for category in CATEGORY_LIST:
        category_df = df[df['category'] == category]
        data_ids = category_df['data_id'].unique()

       
        for data_id in data_ids:
            single_data_df = category_df[category_df['data_id'] == data_id]

          
            original_data = single_data_df[['x', 'y', 'z']].values


            for method_idx, method_config in enumerate(methods):
                for aug_idx in range(augment_per_method):
                    # 딕셔너리를 복사해서 사용 (원본 수정 방지)
                    config_copy = method_config.copy()
                    method = config_copy.pop('method')
                    augmented_data = augment_single_data(original_data, method, **config_copy)

                    # 새로운 data_id 생성 (예: 1 -> 1001, 1002, ...)
                    new_data_id = int(data_id * 1000 + (method_idx * augment_per_method + aug_idx + 1))

                    # DataFrame 형태로 변환
                    augmented_rows = []
                    for time_step, coords in enumerate(augmented_data):
                        augmented_rows.append([
                            category,
                            new_data_id,
                            time_step,
                            float(coords[0]),
                            float(coords[1]),
                            float(coords[2])
                        ])

                    aug_df = pd.DataFrame(
                        augmented_rows,
                        columns=['category', 'data_id', 'time_step', 'x', 'y', 'z']
                    )
                    all_data.append(aug_df)

    
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df


# 파일 저장 함수
def save_augmented_data(df: pd.DataFrame, output_folder: str):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for category in CATEGORY_LIST:
        category_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        category_df = df[df['category'] == category]
        data_ids = category_df['data_id'].unique()

        for data_id in data_ids:
            single_data = category_df[category_df['data_id'] == data_id]

           
            file_path = os.path.join(category_folder, f"{data_id}.txt")

            
            with open(file_path, 'w', encoding='utf-8') as f:
                for idx, row in single_data.iterrows():
                    x, y, z = int(row['x']), int(row['y']), int(row['z'])
                   
                    line = f"r,{idx},0,0/0/0/0,0/0/0/0,0/0/0/0,0/0,0/0/0/0,0/0/0,0/0/0,,{x}/{y}/{z},,#\n"
                    f.write(line)

    print(f"Augmented data saved to: {output_folder}")