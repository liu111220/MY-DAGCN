import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

# 数据配置
signal_size = 1024

# 域映射 (负载 -> 域编号)
DOMAIN_MAP = {0: '0HP', 1: '1HP', 2: '2HP', 3: '3HP'}

# 类别映射 (故障类型 -> 类别编号)
CLASS_MAP = {'normal': 0, 'inner': 1, 'ball': 2, 'outer': 3}


def get_files(root, N):
    """
    加载指定域的所有数据
    root: 数据根目录
    N: 域ID列表，如 [3] 表示加载3HP的数据
    return: [data_list, label_list] - 分段后的样本列表
    """
    data, lab = [], []
    
    for d in tqdm(N, desc="Loading domains"):
        dom = DOMAIN_MAP[int(d)]
        
        for cname, cid in CLASS_MAP.items():
            cdir = os.path.join(root, cname, dom)
            
            if not os.path.isdir(cdir):
                print(f"警告: 目录不存在 {cdir}")
                continue
            
            mat_files = [f for f in os.listdir(cdir) if f.lower().endswith('.mat')]
            
            if len(mat_files) == 0:
                print(f"警告: {cdir} 中没有.mat文件")
                continue
            
            print(f"  加载 {cname}/{dom}: {len(mat_files)} 个文件")
            
            for fn in mat_files:
                filepath = os.path.join(cdir, fn)
                try:
                    data_segments, lab_segments = data_load(filepath, fn, cid)
                    data.extend(data_segments)
                    lab.extend(lab_segments)
                except Exception as e:
                    print(f"  错误: 无法加载 {filepath}: {e}")
                    continue
    
    print(f"总共加载 {len(data)} 个样本")
    return [data, lab]


def data_load(filename, axisname, label):
    """
    从 .mat 文件中读取 DE 通道数据并分段
    filename: .mat 文件路径
    axisname: 文件名
    label: 类别标签
    return: (data_list, label_list)
    """
    # 载入 .mat 文件
    m = loadmat(filename)
    var = None

    # 方法1: 根据文件名推断变量名 (如 X097_DE_time)
    dataname = os.path.splitext(os.path.basename(axisname))[0]
    if dataname.isdigit():
        num = int(dataname)
        if num < 100:
            guess = f"X0{dataname}_DE_time"
        else:
            guess = f"X{dataname}_DE_time"
        if guess in m:
            var = guess

    # 方法2: 自动查找包含 DE 的变量
    if var is None:
        keys = [k for k in m.keys() if not k.startswith('__')]
        # 优先查找同时包含 DE 和 time 的键
        de_time_keys = [k for k in keys if 'DE' in k.upper() and 'TIME' in k.upper()]
        if de_time_keys:
            var = de_time_keys[0]
        else:
            # 退而求其次，查找包含 DE 的键
            de_keys = [k for k in keys if 'DE' in k.upper()]
            if de_keys:
                var = de_keys[0]
    
    if var is None:
        raise KeyError(f"在 {filename} 中未找到 DE 通道数据")

    # 读取并处理数据
    fl = np.asarray(m[var]).squeeze().astype(np.float32)
    if fl.ndim != 1:
        fl = fl.reshape(-1).astype(np.float32)

    # 按 signal_size 分段
    data, lab = [], []
    start, end = 0, signal_size
    total_len = int(fl.shape[0])
    
    while end <= total_len:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


class CWRU(object):
    num_classes = 4  # normal, inner, ball, outer
    inputchannel = 1
    
    def __init__(self, data_dir, transfer_task, normlizetype="mean-std"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # 加载源域数据
            print(f"\n{'='*50}")
            print(f"加载源域: {self.source_N} ({[DOMAIN_MAP[i] for i in self.source_N]})")
            print(f"{'='*50}")
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            
            print("\n源域类别分布:")
            label_counts = data_pd['label'].value_counts().sort_index()
            for label_id, count in label_counts.items():
                class_name = [k for k, v in CLASS_MAP.items() if v == label_id][0]
                print(f"  {class_name} (ID={label_id}): {count} 个样本")
            
            # 检查是否有足够的样本进行分层划分
            min_samples = label_counts.min()
            if min_samples < 2:
                raise ValueError(f"源域中某些类别样本数少于2个，无法进行分层划分。最少样本数: {min_samples}")
            
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"]
            )
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # 加载目标域数据
            print(f"\n{'='*50}")
            print(f"加载目标域: {self.target_N} ({[DOMAIN_MAP[i] for i in self.target_N]})")
            print(f"{'='*50}")
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            
            print("\n目标域类别分布:")
            label_counts = data_pd['label'].value_counts().sort_index()
            for label_id, count in label_counts.items():
                class_name = [k for k, v in CLASS_MAP.items() if v == label_id][0]
                print(f"  {class_name} (ID={label_id}): {count} 个样本")
            
            min_samples = label_counts.min()
            if min_samples < 2:
                raise ValueError(f"目标域中某些类别样本数少于2个。最少样本数: {min_samples}")
            
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"]
            )
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            
            print(f"\n{'='*50}")
            print("数据划分完成:")
            print(f"  源域训练集: {len(source_train)} 样本")
            print(f"  源域验证集: {len(source_val)} 样本")
            print(f"  目标域训练集: {len(target_train)} 样本")
            print(f"  目标域验证集: {len(target_val)} 样本")
            print(f"{'='*50}\n")
            
            return source_train, source_val, target_train, target_val
        
        else:
            # 非迁移学习模式
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"]
            )
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            
            return source_train, source_val, target_val