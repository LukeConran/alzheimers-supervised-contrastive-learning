import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from scipy import ndimage
LABEL_MAP = {
    "Alzheimer's Disease": 0,
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
    "Alzheimer's Disease": 0,
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
