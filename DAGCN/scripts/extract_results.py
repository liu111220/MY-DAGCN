# DAGCN/scripts/extract_results.py
"""
从训练日志中提取最后10个epoch的平均准确率
"""
import os
import re
import sys
from pathlib import Path
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.config import TRANSFER_TASKS, RESULTS_DIR, ANALYSIS_DIR


def extract_accuracies_from_log(log_file):
    """从日志文件提取所有epoch的准确率"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    target_val_accs = []
    epochs = []
    
    for line in lines:
        # 提取target_val准确率
        match = re.search(r'Epoch:\s+(\d+).*target_val-Acc:\s+([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            epochs.append(epoch)
            target_val_accs.append(acc)
    
    return {
        'epochs': epochs,
        'target_val_accs': target_val_accs
    }


def calculate_final_result(accs, last_n=10):
    """计算最后N个epoch的平均结果"""
    if not accs or len(accs['target_val_accs']) < last_n:
        return None
    
    last_n_accs = accs['target_val_accs'][-last_n:]
    
    return {
        'mean': np.mean(last_n_accs),
        'std': np.std(last_n_accs),
        'min': np.min(last_n_accs),
        'max': np.max(last_n_accs),
        'last_n_values': last_n_accs,
        'best_overall': max(accs['target_val_accs']),
        'best_epoch': accs['epochs'][accs['target_val_accs'].index(max(accs['target_val_accs']))]
    }


def extract_all_results():
    """提取所有任务的结果"""
    
    print("="*80)
    print("  提取DAGCN所有任务的训练结果")
    print("="*80)
    
    dagcn_dir = os.path.join(RESULTS_DIR, 'DAGCN')
    
    if not os.path.exists(dagcn_dir):
        print(f"\n错误: 结果目录不存在: {dagcn_dir}")
        return
    
    # 扫描所有任务目录
    task_results = {}
    
    for task_id in TRANSFER_TASKS.keys():
        # 查找匹配的目录（最新的）
        pattern = f"DAGCN_{task_id}_*"
        matching_dirs = sorted(Path(dagcn_dir).glob(pattern))
        
        if not matching_dirs:
            print(f"\n警告: 未找到任务 {task_id} 的结果")
            continue
        
        # 使用最新的目录
        latest_dir = matching_dirs[-1]
        log_file = latest_dir / 'train.log'
        
        print(f"\n处理: {task_id}")
        print(f"  目录: {latest_dir.name}")
        
        # 提取准确率
        accs = extract_accuracies_from_log(log_file)
        
        if accs is None:
            print(f"  ✗ 未找到日志文件")
            continue
        
        # 计算最后10个epoch的结果
        final_result = calculate_final_result(accs, last_n=10)
        
        if final_result is None:
            print(f"  ✗ epoch数量不足")
            continue
        
        task_results[task_id] = {
            'mean_acc': final_result['mean'],
            'std_acc': final_result['std'],
            'best_acc': final_result['best_overall'],
            'best_epoch': final_result['best_epoch'],
            'last_10_accs': final_result['last_n_values'],
            'task_name': TRANSFER_TASKS[task_id]['name'],
            'result_dir': str(latest_dir)
        }
        
        print(f"  ✓ 最后10轮平均准确率: {final_result['mean']*100:.2f}% ± {final_result['std']*100:.2f}%")
        print(f"    历史最佳: {final_result['best_overall']*100:.2f}% (Epoch {final_result['best_epoch']})")
        
        # 保存每个任务的详细结果
        result_file = latest_dir / 'final_results.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"  {task_id}: {TRANSFER_TASKS[task_id]['name']}\n")
            f.write("="*80 + "\n\n")
            f.write(f"最后10个Epoch的平均结果:\n")
            f.write(f"  平均准确率: {final_result['mean']*100:.2f}% ± {final_result['std']*100:.2f}%\n")
            f.write(f"  最小值: {final_result['min']*100:.2f}%\n")
            f.write(f"  最大值: {final_result['max']*100:.2f}%\n\n")
            f.write(f"历史最佳准确率: {final_result['best_overall']*100:.2f}% (Epoch {final_result['best_epoch']})\n\n")
            f.write(f"最后10个Epoch的详细数据:\n")
            for i, acc in enumerate(final_result['last_n_values'], 1):
                f.write(f"  Epoch {len(accs['epochs'])-10+i}: {acc*100:.2f}%\n")
    
    # 保存汇总结果
    summary_csv = os.path.join(ANALYSIS_DIR, 'DAGCN_results_summary.csv')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Task ID', 'Task Name', 'Mean Accuracy (%)', 'Std (%)', 'Best Accuracy (%)', 'Best Epoch'])
        
        for task_id in sorted(task_results.keys()):
            result = task_results[task_id]
            writer.writerow([
                task_id,
                result['task_name'],
                f"{result['mean_acc']*100:.2f}",
                f"{result['std_acc']*100:.2f}",
                f"{result['best_acc']*100:.2f}",
                result['best_epoch']
            ])
        
        # 计算平均值
        if task_results:
            avg_mean = np.mean([r['mean_acc'] for r in task_results.values()])
            writer.writerow(['Average', '', f"{avg_mean*100:.2f}", '', '', ''])
    
    print(f"\n\n{'='*80}")
    print(f"  结果提取完成")
    print(f"{'='*80}")
    print(f"\n共处理: {len(task_results)}/{len(TRANSFER_TASKS)} 个任务")
    print(f"结果已保存到: {summary_csv}")
    
    # 显示汇总表格
    print(f"\n{'='*80}")
    print(f"  DAGCN 结果汇总")
    print(f"{'='*80}\n")
    print(f"{'Task ID':<15} {'Task Name':<15} {'Mean Acc':<12} {'Best Acc':<12}")
    print("-"*80)
    
    for task_id in sorted(task_results.keys()):
        result = task_results[task_id]
        print(f"{task_id:<15} {result['task_name']:<15} {result['mean_acc']*100:>10.2f}% {result['best_acc']*100:>10.2f}%")
    
    if task_results:
        avg_mean = np.mean([r['mean_acc'] for r in task_results.values()])
        print("-"*80)
        print(f"{'Average':<15} {'':<15} {avg_mean*100:>10.2f}%")
    
    print("\n下一步: 所有12个任务完成后，运行 generate_table.py 生成论文表格")


if __name__ == "__main__":
    extract_all_results()