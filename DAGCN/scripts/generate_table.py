# DAGCN/scripts/generate_table.py
"""
生成类似论文Table II的对比表格
"""
import os
import sys
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.config import TRANSFER_TASKS, PAPER_RESULTS, ANALYSIS_DIR


def load_dagcn_results():
    """加载DAGCN的实验结果"""
    csv_file = os.path.join(ANALYSIS_DIR, 'DAGCN_results_summary.csv')
    
    if not os.path.exists(csv_file):
        print(f"错误: 未找到结果文件 {csv_file}")
        print("请先运行 extract_results.py")
        return None
    
    results = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Task ID'] == 'Average':
                results['Average'] = float(row['Mean Accuracy (%)']) / 100
            else:
                task_id = row['Task ID']
                results[task_id] = float(row['Mean Accuracy (%)']) / 100
    
    return results


def generate_comparison_table():
    """生成对比表格"""
    
    print("="*80)
    print("  生成论文格式对比表格")
    print("="*80)
    
    # 加载DAGCN结果
    dagcn_results = load_dagcn_results()
    
    if dagcn_results is None:
        return
    
    # 所有方法（包括论文中的和DAGCN）
    all_methods = list(PAPER_RESULTS.keys()) + ['DAGCN (Ours)']
    
    # 创建完整的结果字典
    all_results = PAPER_RESULTS.copy()
    all_results['DAGCN (Ours)'] = dagcn_results
    
    # 生成Markdown表格
    output_file = os.path.join(ANALYSIS_DIR, 'comparison_table.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 表头
        f.write("# DAGCN Results Comparison\n\n")
        f.write("## Table: Cross-domain Fault Diagnosis Results\n\n")
        
        # 表格头
        header = "| Method | "
        header += " | ".join([TRANSFER_TASKS[task_id]['name'] for task_id in sorted(TRANSFER_TASKS.keys())])
        header += " | Average |\n"
        
        separator = "|" + "---|" * (len(TRANSFER_TASKS) + 2) + "\n"
        
        f.write(header)
        f.write(separator)
        
        # 每个方法的结果
        for method in all_methods:
            if method not in all_results:
                continue
            
            row = f"| {method} |"
            results = all_results[method]
            
            for task_id in sorted(TRANSFER_TASKS.keys()):
                if task_id in results:
                    acc = results[task_id]
                    row += f" {acc*100:.2f}% |"
                else:
                    row += " - |"
            
            # 平均值
            if 'Average' in results:
                row += f" **{results['Average']*100:.2f}%** |"
            else:
                task_accs = [results[task_id] for task_id in sorted(TRANSFER_TASKS.keys()) if task_id in results]
                if task_accs:
                    avg = np.mean(task_accs)
                    row += f" **{avg*100:.2f}%** |"
                else:
                    row += " - |"
            
            f.write(row + "\n")
        
        # 添加说明
        f.write("\n## Notes\n\n")
        f.write("- Results are shown as accuracy (%)\n")
        f.write("- DAGCN (Ours): Average of last 10 epochs\n")
        f.write("- Other methods: Results from original papers\n")
    
    # 也生成纯文本版本
    txt_output = os.path.join(ANALYSIS_DIR, 'comparison_table.txt')
    
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("  Cross-domain Fault Diagnosis Results Comparison\n")
        f.write("="*120 + "\n\n")
        
        # 表头
        header = f"{'Method':<20}"
        for task_id in sorted(TRANSFER_TASKS.keys()):
            header += f"{TRANSFER_TASKS[task_id]['name']:>8} "
        header += f"{'Average':>10}"
        f.write(header + "\n")
        f.write("-"*120 + "\n")
        
        # 每个方法的结果
        for method in all_methods:
            if method not in all_results:
                continue
            
            row = f"{method:<20}"
            results = all_results[method]
            
            for task_id in sorted(TRANSFER_TASKS.keys()):
                if task_id in results:
                    acc = results[task_id]
                    row += f"{acc*100:>7.2f}% "
                else:
                    row += f"{'  -':>8} "
            
            # 平均值
            if 'Average' in results:
                row += f"{results['Average']*100:>9.2f}%"
            else:
                task_accs = [results[task_id] for task_id in sorted(TRANSFER_TASKS.keys()) if task_id in results]
                if task_accs:
                    avg = np.mean(task_accs)
                    row += f"{avg*100:>9.2f}%"
                else:
                    row += f"{'  -':>10}"
            
            f.write(row + "\n")
    
    print(f"\n✓ Markdown表格已保存到: {output_file}")
    print(f"✓ 文本表格已保存到: {txt_output}")
    
    # 在控制台显示
    print(f"\n{'='*120}")
    print(f"  对比结果")
    print(f"{'='*120}\n")
    
    with open(txt_output, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == "__main__":
    generate_comparison_table()