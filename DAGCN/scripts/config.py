# DAGCN/scripts/config.py
"""
DAGCN实验配置文件
定义所有的迁移任务和实验参数
"""

# 数据路径
DATA_DIR = r"D:\桌面\CWRU_12K_DE"
RESULTS_DIR = r"D:\桌面\DAGCN-main\results"
ANALYSIS_DIR = r"D:\桌面\DAGCN-main\analysis"

# 所有12个迁移任务
TRANSFER_TASKS = {
    'Task_0to1': {'source': [0], 'target': [1], 'name': '0HP→1HP'},
    'Task_0to2': {'source': [0], 'target': [2], 'name': '0HP→2HP'},
    'Task_0to3': {'source': [0], 'target': [3], 'name': '0HP→3HP'},
    
    'Task_1to0': {'source': [1], 'target': [0], 'name': '1HP→0HP'},
    'Task_1to2': {'source': [1], 'target': [2], 'name': '1HP→2HP'},
    'Task_1to3': {'source': [1], 'target': [3], 'name': '1HP→3HP'},
    
    'Task_2to0': {'source': [2], 'target': [0], 'name': '2HP→0HP'},
    'Task_2to1': {'source': [2], 'target': [1], 'name': '2HP→1HP'},
    'Task_2to3': {'source': [2], 'target': [3], 'name': '2HP→3HP'},
    
    'Task_3to0': {'source': [3], 'target': [0], 'name': '3HP→0HP'},
    'Task_3to1': {'source': [3], 'target': [1], 'name': '3HP→1HP'},
    'Task_3to2': {'source': [3], 'target': [2], 'name': '3HP→2HP'},
}

# 训练参数（与论文一致）
TRAIN_CONFIG = {
    'model_name': 'DAGCN_features',
    'batch_size': 64,
    'max_epoch': 300,
    'middle_epoch': 50,
    'lr': 0.001,
    'cuda_device': '0',
    'bottleneck': True,
    'bottleneck_num': 256,
    'domain_adversarial': True,
    'hidden_size': 1024,
    'normlizetype': 'mean-std',
    'last_batch': False,
}

# 论文中其他方法的结果（Table II）
# 这些是示例数据，需要根据论文实际数据填写
PAPER_RESULTS = {
    'Baseline': {
        'Task_0to1': 0.6192, 'Task_0to2': 0.5337, 'Task_0to3': 0.3616,
        'Task_1to0': 0.8100, 'Task_1to2': 0.6975, 'Task_1to3': 0.5419,
        'Task_2to0': 0.7537, 'Task_2to1': 0.9442, 'Task_2to3': 0.6787,
        'Task_3to0': 0.5490, 'Task_3to1': 0.6787, 'Task_3to2': 0.8020,
        'Average': 0.6792,
    },
    'CORAL': {
        'Task_0to1': 0.6624, 'Task_0to2': 0.5459, 'Task_0to3': 0.3828,
        'Task_1to0': 0.9200, 'Task_1to2': 0.7568, 'Task_1to3': 0.6439,
        'Task_2to0': 0.9044, 'Task_2to1': 0.8142, 'Task_2to3': 0.8742,
        'Task_3to0': 0.8025, 'Task_3to1': 0.9180, 'Task_3to2': 0.9262,
        'Average': 0.7537,
    },
    # 添加其他方法的结果...
}