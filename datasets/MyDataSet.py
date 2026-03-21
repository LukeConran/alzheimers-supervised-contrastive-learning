import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from scipy import ndimage
LABEL_MAP = {
    "Dementia": 0,
    "Mild Cognitive Impairment": 1,
    "Normal Cognition": 2,
}
KEY_MAPPING_atrophy = {
    "Amygdala": "amygdala",
    "medial temporal lobe": "medial_temporal",
    "fusiform": "fusiform",
    "precuneus": "precuneus",
    "superior parietal": "superior_parietal",
    "medial temporal lobe (vs cortex)": "medial_temporal_vs_cortex",
    "parietal lobe (vs cortex)": "parietal_vs_cortex",
    "frontal lobe": "frontal_lobe",
    "temporal lobe": "temporal_lobe",
    "parietal lobe": "parietal_lobe",
    "occipital lobe": "occipital_lobe",
    "Overall (cortex)": "overall",
    "Ventricle enlargement": "ventricle_enlargement",
    "Lateral Ventricle Temporal shape": "ventricle_temporal_shape",
    "Lateral Ventricle Frontal shape": "ventricle_frontal_shape",
    "Vascular disease": "vascular_disease",
    "hippocampal": "hippocampal",
    "entorhinal": "entorhinal",
    "parahippocampal": "parahippocampal"
}
LABEL_MAP_test = {
    "Dementia": 0,
    "Mild Cognitive Impairment": 1,
    "Normal Cognition": 2,
}
def normalize_label(text: str) -> str:
    # 把所有右单引号 U+2019 替换成 ASCII 单引号 '
    return text.replace("\u2019", "'").strip()

class MyDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        json_path: 包含{id, label}的JSON文件路径
        image_dir: 所有 .npz 图像文件所在目录（文件名为 0.npz, 1.npz,...）
        transform: 图像预处理函数（如需添加）
        """
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # ✅ 只保留图像文件存在的样本
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"⚠️ Skipping missing image: {item['id']}.npz")

        # 按 id 排序（可选）
        self.data = sorted(self.data, key=lambda x: x["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]  # shape: (D, H, W)
        image = self.__data_process__(image)
        image_tensor = torch.from_numpy(image).float()

        label_text = item["label"]
        label = LABEL_MAP[label_text]

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label
    
    def __data_process__(self, data): 

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
    def __resize_data__(self, data):
        '''
        Resize the data to the input size
        ''' 

        [channel, depth, height, width] = data.shape
        scale = [channel,79*1.0/depth, 95*1.0/height, 79*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

class MyDataset_atrophy(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        json_path: 包含{id, label}的JSON文件路径
        image_dir: 所有 .npz 图像文件所在目录（文件名为 0.npz, 1.npz,...）
        transform: 图像预处理函数（如需添加）
        """
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # ✅ 只保留图像文件存在的样本
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"⚠️ Skipping missing image: {item['id']}.npz")

        # 按 id 排序（可选）
        self.data = sorted(self.data, key=lambda x: x["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]  # shape: (D, H, W)
        image = self.__data_process__(image)
        image_tensor = torch.from_numpy(image).float()
        labels_dict = {}
        label_text = normalize_label(item["label"])
        label = LABEL_MAP[label_text]
        labels_dict["disease"] = torch.tensor(label, dtype=torch.long)
        for key, value in item.items():
            if key in ["id", "label"]:  # 跳过非标签字段
                continue
            labels_dict[key] = torch.tensor(int(value), dtype=torch.long)      

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, labels_dict
    
    def __data_process__(self, data): 

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

class MyDataset_atrophy_test(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        json_path: 包含{id, label}的JSON文件路径
        image_dir: 所有 .npz 图像文件所在目录（文件名为 0.npz, 1.npz,...）
        transform: 图像预处理函数（如需添加）
        """
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # ✅ 只保留图像文件存在的样本
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"⚠️ Skipping missing image: {item['id']}.npz")

        # 按 id 排序（可选）
        self.data = sorted(self.data, key=lambda x: x["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- 加载图像 ---
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]  # (D, H, W)

        # 归一化 & 加 channel 维
        image = self.__data_process__(image)
        image_tensor = torch.from_numpy(image).float()

        # --- 构造标签字典 ---
        labels_dict = {}
        # 疾病分类
        labels_dict["disease"] = torch.tensor(LABEL_MAP[item["label"]], dtype=torch.long)

        # 其它 heads
        for key, value in item.items():
            if key in ["id", "label"]:
                continue
            if key not in KEY_MAPPING_atrophy:
                print(f"⚠️ Warning: {key} not in KEY_MAPPING, skipping")
                continue
            mapped_key = KEY_MAPPING_atrophy[key]
            labels_dict[mapped_key] = torch.tensor(int(value), dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, labels_dict
    
    def __data_process__(self, data): 

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

class MyDataset_test(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        """
        json_path: 包含{id, label}的JSON文件路径
        image_dir: 所有 .npz 图像文件所在目录（文件名为 0.npz, 1.npz,...）
        transform: 图像预处理函数（如需添加）
        """
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # ✅ 只保留图像文件存在的样本
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"⚠️ Skipping missing image: {item['id']}.npz")

        # 按 id 排序（可选）
        self.data = sorted(self.data, key=lambda x: x["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]  # shape: (D, H, W)
        image = self.__data_process__(image)
        image_tensor = torch.from_numpy(image).float()

        label_text = item["label"]
        label = LABEL_MAP_test[label_text]

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label
    
    def __data_process__(self, data): 

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
    def __resize_data__(self, data):
        '''
        Resize the data to the input size
        ''' 

        [channel, depth, height, width] = data.shape
        scale = [channel,79*1.0/depth, 95*1.0/height, 79*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

# ═══════════════════════════════════════════════════════════════════════════════
# ContrastiveDataset
# ═══════════════════════════════════════════════════════════════════════════════
# Extends MyDataset to support contrastive learning.
#
# Key difference from MyDataset:
#   __getitem__ returns TWO independently-augmented views of the same MRI scan
#   instead of one. The SupCon loss uses these pairs to learn that same-class
#   scans should cluster together in embedding space.
#
# Augmentation strategy for 3D MRI:
#   We keep augmentations conservative because:
#     (1) Brain anatomy is spatially structured — aggressive crops can remove
#         disease-relevant regions (hippocampus, amygdala, etc.).
#     (2) MRI intensities are already normalized — color jitter is inappropriate.
#   Safe augmentations used here:
#     - Random axis flips (left-right, anterior-posterior, superior-inferior)
#     - Gaussian noise (simulates scanner noise variability)
#     - Intensity scaling (simulates gain/contrast variability across scanners)
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveDataset(MyDataset):
    """
    Dataset for supervised contrastive learning.

    Each call to __getitem__ loads one MRI scan and returns two independently
    augmented views (view1, view2) along with the disease class label.

    The two views are fed separately through the encoder, and their embeddings
    are concatenated before being passed to SupConLoss.

    Args:
        json_path (str):    Path to JSON file with {id, label} entries.
        image_dir (str):    Directory containing <id>.npz image files.
        noise_std (float):  Std of Gaussian noise added during augmentation.
                            Default 0.05 adds subtle noise relative to the
                            z-scored volume (mean≈0, std≈1).
        scale_range (tuple): Min/max multiplicative intensity scaling factor.
                             (0.9, 1.1) means ±10% brightness variation.
    """

    def __init__(self, json_path, image_dir, noise_std=0.05, scale_range=(0.9, 1.1)):
        super().__init__(json_path, image_dir)
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{item['id']}.npz")
        image_npz = np.load(image_path)
        image = image_npz["image_mr"]               # (D, H, W), raw volume
        image = self.__data_process__(image)         # intensity normalization (from parent)

        # Apply two DIFFERENT random augmentations to the same normalized volume.
        # Each call to _augment samples new random values, so view1 ≠ view2.
        view1 = self._augment(image)
        view2 = self._augment(image)

        label = LABEL_MAP[item["label"]]

        return (
            torch.from_numpy(view1).float(),    # augmented view 1: (D, H, W)
            torch.from_numpy(view2).float(),    # augmented view 2: (D, H, W)
            label,                              # integer class index
        )

    def _augment(self, volume):
        """
        Apply a random combination of MRI-safe 3D augmentations.

        Each augmentation is applied independently with 50% probability,
        so the two views returned by __getitem__ will almost always differ.

        Args:
            volume (np.ndarray): Normalized 3D volume, shape (D, H, W).

        Returns:
            np.ndarray: Augmented volume, same shape as input.
        """
        volume = volume.copy()   # avoid mutating the shared numpy array

        # ── Augmentation 1: Random axis flips ────────────────────────────────
        # Flipping along each axis is anatomically plausible for MRI
        # (left-right symmetry of the brain is well-established).
        for axis in range(3):
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis).copy()

        # ── Augmentation 2: Additive Gaussian noise ───────────────────────────
        # Simulates thermal/scanner noise. Kept small (noise_std=0.05) relative
        # to the z-scored volume so disease-relevant signal is preserved.
        if np.random.rand() > 0.5:
            volume = volume + np.random.normal(0, self.noise_std, volume.shape)

        # ── Augmentation 3: Intensity scaling ────────────────────────────────
        # Simulates gain variation between MRI scanners or scan sessions.
        # Only applied to non-zero voxels to avoid amplifying background noise.
        if np.random.rand() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            volume[volume != 0] *= scale

        return volume


class MyDataset_org(Dataset):
    
    def __init__(self, datas=None, labels=None, shape=None, input_D=None, input_H=None, input_W=None, phase='train', transforms=None):
        self.datas = datas
        self.labels = labels
        self.transforms = transforms
        self.shape = shape
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase

    #返回整个数据集大小
    def __len__(self):
        return self.datas.shape[0]
    
    #根据索引index返回dataset[index]
    def __getitem__(self,index):
        if self.phase == 'train':
            img = self.__data_process__(self.datas[index])
            label = self.labels[index]
            img = torch.tensor(img)
            if self.transforms:
                img = self.transforms(img)
            return img,label
        elif self.phase == 'test':
            img = self.__data_process__(self.datas[index])
            img = torch.tensor(img)
            if self.transforms:
                img = self.transforms(img)
            return img
    
    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        '''
        Resize the data to the input size
        ''' 
        if self.shape == 2:
            [depth, height, width] = data.shape
            scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        else:
            [channel, depth, height, width] = data.shape
            scale = [channel,self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    
    def __data_process__(self, data): 

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
