# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 00:13:45 2026

@author: shiqi
"""

"""
generate_simple_report.py
生成简洁的数据分析报告 - 修复版
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

def generate_simple_report():
    """生成简洁的报告"""
    
    print("正在生成数据分析报告...")
    
    # 1. 读取CSV文件
    try:
        train_df = pd.read_csv('coco_person_car_train_local.csv')
        val_df = pd.read_csv('coco_person_car_val_local.csv')
    except FileNotFoundError as e:
        print(f" 找不到CSV文件: {e}")

        return
    
    # 2. 基础统计
    print(" 统计数据集信息...")
    
    train_total = len(train_df)
    val_total = len(val_df)
    total_images = train_total + val_total
    
    train_person = len(train_df[train_df['label'] == 0])
    train_car = len(train_df[train_df['label'] == 1])
    val_person = len(val_df[val_df['label'] == 0])
    val_car = len(val_df[val_df['label'] == 1])
    
    # 3. 生成可视化
    print(" 生成可视化图表...")
    
    # 创建图表目录
    os.makedirs('visualizations', exist_ok=True)
    
    # 图表1: 整体分布
    plt.figure(figsize=(10, 5))
    
    # 子图1: 训练集分布
    plt.subplot(1, 2, 1)
    labels = ['Person', 'Car']
    sizes = [train_person, train_car]
    colors = ['#ff9999', '#66b3ff']  # 修正颜色代码
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Train Set\nTotal: {train_total} images')
    
    # 子图2: 验证集分布
    plt.subplot(1, 2, 2)
    sizes = [val_person, val_car]
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Validation Set\nTotal: {val_total} images')
    
    plt.tight_layout()
    plt.savefig('visualizations/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图表2: 对比图
    plt.figure(figsize=(8, 6))
    
    categories = ['Train Set', 'Validation Set']
    person_counts = [train_person, val_person]
    car_counts = [train_car, val_car]
    
    x = range(len(categories))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], person_counts, width, label='Person', color='#ff9999')
    plt.bar([i + width/2 for i in x], car_counts, width, label='Car', color='#66b3ff')  # 修正颜色代码
    
    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.title('Person vs Car Distribution')
    plt.xticks(x, categories)
    plt.legend()
    
    # 添加数值标签
    for i, (p, c) in enumerate(zip(person_counts, car_counts)):
        plt.text(i - width/2, p + max(person_counts+car_counts)*0.01, str(p), ha='center')
        plt.text(i + width/2, c + max(person_counts+car_counts)*0.01, str(c), ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 生成Markdown报告
    print("📝 生成报告文件...")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# COCO Person-Car Dataset Analysis Report

**Generated on:** {current_time}

## 📊 Dataset Statistics

### Image Counts
| Dataset | Total Images | Person Images | Car Images |
|---------|--------------|---------------|------------|
| **Train Set** | {train_total:,} | {train_person:,} | {train_car:,} |
| **Validation Set** | {val_total:,} | {val_person:,} | {val_car:,} |
| **Total** | {total_images:,} | {train_person + val_person:,} | {train_car + val_car:,} |

### Percentage Distribution
- **Train Set:** Person {train_person/train_total*100:.1f}% | Car {train_car/train_total*100:.1f}%
- **Validation Set:** Person {val_person/val_total*100:.1f}% | Car {val_car/val_total*100:.1f}%
- **Overall:** Person {(train_person + val_person)/total_images*100:.1f}% | Car {(train_car + val_car)/total_images*100:.1f}%


##  Dataset Information

### File Details
- **Train CSV:** `coco_person_car_train_local.csv` ({train_total} records)
- **Validation CSV:** `coco_person_car_val_local.csv` ({val_total} records)
- **Images stored locally in:**
  - `coco/images/person_car_train2017/`
  - `coco/images/person_car_val2017/`


---
*Report generated automatically*
"""
    
    # 保存报告
    with open('data_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已生成: data_analysis.md")
    print(f"✅ 可视化图表已保存到: visualizations/")

def check_files():
    """检查文件是否存在"""
    required_files = [
        'coco_person_car_train_local.csv',
        'coco_person_car_val_local.csv'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return missing

def main():
    print("=" * 50)
    print("COCO Person-Car Dataset Analysis Report Generator")
    print("=" * 50)
    
    # 检查文件
    missing_files = check_files()
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请先运行下载脚本:")
        print("   python download_coco_simple.py")
        return
    
    # 生成报告
    generate_simple_report()
    
    print("\n" + "=" * 50)
    print(" 数据分析报告生成完成!")
    print("=" * 50)
    print("\n生成的文件:")
    print("    data_analysis.md")
    print("     visualizations/data_distribution.png")
    print("     visualizations/comparison.png")

if __name__ == "__main__":
    # 安装必要依赖
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("正在安装依赖...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib"])
    
    main()