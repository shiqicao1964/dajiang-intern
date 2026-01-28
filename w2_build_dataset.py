"""
build_dataset.py
构建COCO人车单一标签分类数据集
"""

import json
import pandas as pd
from pycocotools.coco import COCO
import os

def create_single_label_dataset(annotation_file, output_csv, split='train'):
    """
    创建单一标签数据集（方案A）
    
    逻辑：
    - 只有人 → label=0
    - 只有车 → label=1
    - 同时有人车 → 舍弃
    """
    print(f"正在处理 {split} 数据集: {annotation_file}")
    
    # 加载COCO标注
    coco = COCO(annotation_file)
    
    # 获取类别ID
    person_id = coco.getCatIds(['person'])[0]
    car_id = coco.getCatIds(['car'])[0]
    
    dataset_records = []
    
    print(f"总图片数: {len(coco.imgs)}")
    print("开始筛选图片...")
    
    # 遍历所有图片标注
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        
        # 获取该图片的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 统计该图片的类别（只关注人和车）
        categories = set()
        for ann in anns:
            if ann['category_id'] in [person_id, car_id]:
                categories.add(ann['category_id'])
        
        # 判断图片类别
        has_person = person_id in categories
        has_car = car_id in categories
        
        # 方案A：单一标签，舍弃混合图片
        if has_person and not has_car:
            # 只有人
            dataset_records.append({
                'file_path': f"./coco/images/person_car_{split}2017/{img_info['file_name']}",  # 相对路径
                'label': 0,  # 人
                'image_id': img_id,
                'split': split,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'coco_url': img_info['coco_url']  #用于后续下载
            })
        elif has_car and not has_person:
            # 只有车
            dataset_records.append({
                'file_path': f"./coco/images/person_car_{split}2017/{img_info['file_name']}",
                'label': 1,  # 车
                'image_id': img_id,
                'split': split,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'coco_url': img_info['coco_url']  # 用于后续下载
            })
        # 其他情况（两者都有或都没有）舍弃
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(dataset_records)
    df.to_csv(output_csv, index=False)
    
    print(f"\n数据集统计 ({split}):")
    print(f"  总图片数: {len(coco.imgs)}")
    print(f"  筛选后图片数: {len(df)}")
    print(f"  只有人的图片 (label=0): {len(df[df['label'] == 0])}")
    print(f"  只有车的图片 (label=1): {len(df[df['label'] == 1])}")
    print(f"  舍弃的图片数: {len(coco.imgs) - len(df)}")
    print(f"  筛选率: {len(df)/len(coco.imgs)*100:.2f}%")
    
    return df

def analyze_dataset_details(coco, person_id, car_id):
    """分析数据集分布"""
    stats = {
        'total': len(coco.imgs),
        'person_only': 0,
        'car_only': 0,
        'both': 0,
        'neither': 0
    }
    
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 统计该图片的类别（只关注人和车）
        categories = set()
        for ann in anns:
            if ann['category_id'] in [person_id, car_id]:
                categories.add(ann['category_id'])
        
        has_person = person_id in categories
        has_car = car_id in categories
        
        if has_person and not has_car:
            stats['person_only'] += 1
        elif has_car and not has_person:
            stats['car_only'] += 1
        elif has_person and has_car:
            stats['both'] += 1
        else:
            stats['neither'] += 1
    
    return stats

def main():
    """主函数"""
    print("COCO人车单一标签数据集构建工具")
    
    # 检查标注文件是否存在
    train_ann = "coco/annotations/instances_train2017.json"
    val_ann = "coco/annotations/instances_val2017.json"
    
    print("\n1. 数据集分析")
    # 分析训练集
    coco_train = COCO(train_ann)
    person_id = coco_train.getCatIds(['person'])[0]
    car_id = coco_train.getCatIds(['car'])[0]
    
    train_stats = analyze_dataset_details(coco_train, person_id, car_id)
    print(f"训练集:")
    print(f"  总图片数: {train_stats['total']}")
    print(f"  只有人的图片: {train_stats['person_only']}")
    print(f"  只有车的图片: {train_stats['car_only']}")
    print(f"  人车都有的图片: {train_stats['both']}")
    print(f"  都没有的图片: {train_stats['neither']}")
    
    # 分析验证集
    coco_val = COCO(val_ann)
    val_stats = analyze_dataset_details(coco_val, person_id, car_id)
    print(f"\n验证集:")
    print(f"  总图片数: {val_stats['total']}")
    print(f"  只有人的图片: {val_stats['person_only']}")
    print(f"  只有车的图片: {val_stats['car_only']}")
    print(f"  人车都有的图片: {val_stats['both']}")
    print(f"  都没有的图片: {val_stats['neither']}")
    

    # 构建训练集
    print("\n构建人车训练集")
    train_df = create_single_label_dataset(
        annotation_file=train_ann,
        output_csv='coco_person_car_train.csv',
        split='train'
    )
    
    # 构建验证集
    print("\n构建人车验证集")
    val_df = create_single_label_dataset(
        annotation_file=val_ann,
        output_csv='coco_person_car_val.csv',
        split='val'
    )
    
    # 总结
    print("数据集构建总结!")
    print(f"训练集: {len(train_df)} 张图片")
    print(f"  标签0 (人): {len(train_df[train_df['label'] == 0])}")
    print(f"  标签1 (车): {len(train_df[train_df['label'] == 1])}")
    print(f"\n验证集: {len(val_df)} 张图片")
    print(f"  标签0 (人): {len(val_df[val_df['label'] == 0])}")
    print(f"  标签1 (车): {len(val_df[val_df['label'] == 1])}")
    print(f"\n总计: {len(train_df) + len(val_df)} 张图片")
   
  

if __name__ == "__main__":
    main()