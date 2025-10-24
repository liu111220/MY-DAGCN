#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
批量训练DAGCN的所有12个迁移任务
"""
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import TRANSFER_TASKS, TRAIN_CONFIG, DATA_DIR, RESULTS_DIR


def train_single_task(task_id, task_config, model_name='DAGCN'):
    """训练单个任务"""
    
    # Create model parent directory with robust error handling
    task_dir = os.path.join(RESULTS_DIR, model_name)
    try:
        # Ensure RESULTS_DIR exists first
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR, exist_ok=True)
            print(f"Created results directory: {RESULTS_DIR}")
        
        # Then create task_dir
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)
            print(f"Created task directory: {task_dir}")
    except Exception as e:
        print(f"Error creating directory {task_dir}: {e}")
        raise
    
    
    # 构建训练命令
    # 传递task_id参数，让train_advanced.py创建规范命名的目录
    cmd = [
        'python', 'train_advanced.py',
        '--model_name', TRAIN_CONFIG['model_name'],
        '--data_dir', DATA_DIR,
        '--transfer_task', f"[{task_config['source']},{task_config['target']}]",
        '--checkpoint_dir', task_dir,  # 传递父目录
        '--task_id', task_id,  # 新增：传递任务ID
        '--batch_size', str(TRAIN_CONFIG['batch_size']),
        '--max_epoch', str(TRAIN_CONFIG['max_epoch']),
        '--middle_epoch', str(TRAIN_CONFIG['middle_epoch']),
        '--lr', str(TRAIN_CONFIG['lr']),
        '--cuda_device', TRAIN_CONFIG['cuda_device'],
        '--bottleneck', str(TRAIN_CONFIG['bottleneck']),
        '--bottleneck_num', str(TRAIN_CONFIG['bottleneck_num']),
        '--domain_adversarial', str(TRAIN_CONFIG['domain_adversarial']),
        '--hidden_size', str(TRAIN_CONFIG['hidden_size']),
        '--normlizetype', TRAIN_CONFIG['normlizetype'],
    ]
    
    print(f"\n{'='*80}")
    print(f"  开始训练: {task_id} ({task_config['name']})")
    print(f"{'='*80}")
    print(f"结果将保存为: DAGCN_{task_id}_YYYYMMDD_HHMMSS")
    print(f"命令: {' '.join(cmd)}\n")
    
    # 执行训练
    start_time = time.time()
    
    try:
        # 切换到DAGCN目录
        os.chdir(Path(__file__).parent.parent)
        
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        elapsed_time = time.time() - start_time
        
        # 找到实际创建的目录（最新的以 DAGCN_task_id 开头的）
        import glob
        pattern = os.path.join(task_dir, f'DAGCN_{task_id}_*')
        matching_dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        if matching_dirs:
            actual_save_dir = matching_dirs[-1]  # 最新的目录
            
            # 保存任务配置
            config_file = os.path.join(actual_save_dir, 'task_config.txt')
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write(f"  任务配置信息\n")
                f.write("="*60 + "\n\n")
                f.write(f"模型: DAGCN\n")
                f.write(f"任务ID: {task_id}\n")
                f.write(f"任务名称: {task_config['name']}\n")
                f.write(f"源域: {task_config['source']} HP\n")
                f.write(f"目标域: {task_config['target']} HP\n\n")
                f.write(f"训练时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)\n")
                f.write(f"训练状态: 成功完成\n\n")
                f.write("训练参数:\n")
                f.write("-"*60 + "\n")
                for key, value in TRAIN_CONFIG.items():
                    f.write(f"  {key}: {value}\n")
            
            print(f"\n✓ {task_id} 训练完成！用时: {elapsed_time/60:.2f} 分钟")
            print(f"  结果保存在: {actual_save_dir}")
            return True, actual_save_dir
        else:
            print(f"\n✗ 未找到训练结果目录")
            return False, None
            
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {task_id} 训练失败！用时: {elapsed_time/60:.2f} 分钟")
        print(f"错误信息: {str(e)}")
        return False, None


def main():
    """主函数：按顺序训练所有任务"""
    
    print("="*80)
    print("  DAGCN 批量训练系统")
    print("="*80)
    print(f"\n总任务数: {len(TRANSFER_TASKS)}")
    print(f"预计总时间: {len(TRANSFER_TASKS) * 4} 分钟 (每个任务约4分钟)")
    print(f"\n结果保存格式: DAGCN_Task_XtoY_YYYYMMDD_HHMMSS")
    print(f"保存位置: {RESULTS_DIR}/DAGCN/\n")
    
    # 显示所有任务
    print("将要训练的任务:")
    print("-"*80)
    for i, (task_id, task_config) in enumerate(TRANSFER_TASKS.items(), 1):
        print(f"  {i:2d}. {task_id:<15} {task_config['name']}")
    print("-"*80)
    
    input("\n按回车键开始训练...")
    
    # 记录训练结果
    results = {}
    completed_tasks = []
    failed_tasks = []
    
    total_start_time = time.time()
    
    # 按顺序训练每个任务
    for i, (task_id, task_config) in enumerate(TRANSFER_TASKS.items(), 1):
        print(f"\n\n{'#'*80}")
        print(f"  进度: {i}/{len(TRANSFER_TASKS)}")
        print(f"{'#'*80}")
        
        success, save_dir = train_single_task(task_id, task_config)
        
        results[task_id] = {
            'success': success,
            'save_dir': save_dir,
            'task_name': task_config['name']
        }
        
        if success:
            completed_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)
        
        # 显示进度
        print(f"\n当前进度: 完成 {len(completed_tasks)}/{len(TRANSFER_TASKS)} 个任务")
        if failed_tasks:
            print(f"失败任务: {', '.join(failed_tasks)}")
    
    # 总结
    total_time = time.time() - total_start_time
    
    print(f"\n\n{'='*80}")
    print(f"  训练完成总结")
    print(f"{'='*80}")
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    print(f"成功: {len(completed_tasks)}/{len(TRANSFER_TASKS)} 个任务")
    
    if completed_tasks:
        print(f"\n✓ 完成的任务:")
        for task_id in completed_tasks:
            print(f"  - {task_id}: {results[task_id]['task_name']}")
            if results[task_id]['save_dir']:
                print(f"    → {os.path.basename(results[task_id]['save_dir'])}")
    
    if failed_tasks:
        print(f"\n✗ 失败的任务:")
        for task_id in failed_tasks:
            print(f"  - {task_id}: {results[task_id]['task_name']}")
    
    # 保存训练总结
    summary_file = os.path.join(RESULTS_DIR, 'DAGCN', 'training_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  DAGCN 训练总结\n")
        f.write("="*80 + "\n\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总用时: {total_time/60:.2f} 分钟\n")
        f.write(f"成功任务: {len(completed_tasks)}/{len(TRANSFER_TASKS)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("任务详情:\n")
        f.write("-"*80 + "\n\n")
        
        for task_id, result in results.items():
            f.write(f"{task_id}: {result['task_name']}\n")
            f.write(f"  状态: {'成功' if result['success'] else '失败'}\n")
            if result['save_dir']:
                f.write(f"  目录: {os.path.basename(result['save_dir'])}\n")
            f.write("\n")
    
    print(f"\n训练总结已保存到: {summary_file}")
    
    if len(completed_tasks) == len(TRANSFER_TASKS):
        print("\n🎉 所有任务训练完成！")
        print("\n下一步：运行以下命令提取结果")
        print("  cd D:\\桌面\\DAGCN-main\\DAGCN")
        print("  python scripts/extract_results.py")
    else:
        print("\n⚠️  部分任务失败，请检查错误信息")


if __name__ == "__main__":
    main()