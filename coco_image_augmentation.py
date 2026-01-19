import os
import random
import json
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from tqdm import tqdm

def create_augmented_dataset(image_dir, annotation_file, output_dir, num_samples=None, seed=42):
    """创建数据增强数据集（随机水平翻转+随机裁剪）"""
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # 加载原始数据
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    
    # 如果指定了样本数量
    if num_samples:
        image_ids = image_ids[:num_samples]
    
    # 准备新标注数据
    new_annotations = {
        "info": coco.dataset.get('info', {}),
        "licenses": coco.dataset.get('licenses', []),
        "images": [],
        "annotations": [],
        "categories": coco.dataset.get('categories', [])
    }
    
    new_image_id = 100000  # 新图像ID起始值
    new_ann_id = 100000    # 新标注ID起始值
    target_size = 224      # 目标尺寸
    
    print(f"开始数据增强处理，共 {len(image_ids)} 张图像...")
    
    for img_id in tqdm(image_ids, desc="Processing images"):
        # 原始图像信息
        img_info = coco.loadImgs([img_id])[0]
        
        # 加载原始图像
        img_path = f"{image_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 1. 随机水平翻转（50%概率）
        do_flip = random.random() < 0.5
        if do_flip:
            image = ImageOps.mirror(image)
        
        # 2. 随机裁剪到目标尺寸
        # 计算最大可能的裁剪区域
        crop_x = random.randint(0, max(0, orig_w - target_size))
        crop_y = random.randint(0, max(0, orig_h - target_size))
        
        # 确保裁剪区域有效
        crop_w = min(target_size, orig_w - crop_x)
        crop_h = min(target_size, orig_h - crop_y)
        
        # 执行裁剪
        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
        # 3. 调整到目标尺寸（如果裁剪尺寸小于目标尺寸）
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 保存增强后的图像
        new_filename = f"aug_{img_info['file_name']}"
        image.save(f"{output_dir}/images/{new_filename}")
        
        # 添加到新图像列表
        new_img_info = {
            "id": new_image_id,
            "width": target_size,
            "height": target_size,
            "file_name": new_filename,
            "license": img_info.get('license', 0),
            "flickr_url": img_info.get('flickr_url', ''),
            "coco_url": img_info.get('coco_url', ''),
            "date_captured": img_info.get('date_captured', ''),
            "_original_id": img_id
        }
        new_annotations["images"].append(new_img_info)
        
        # 处理原始标注
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)
        
        for ann in annotations:
            new_ann = ann.copy()
            
            # 更新ID
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = new_image_id
            
            # 调整边界框
            if 'bbox' in new_ann:
                x, y, w, h = new_ann['bbox']
                
                # 1. 应用水平翻转
                if do_flip:
                    x = orig_w - x - w
                
                # 2. 应用裁剪偏移
                x = x - crop_x
                y = y - crop_y
                
                # 3. 确保bbox在裁剪区域内
                x = max(0, min(x, target_size - 1))
                y = max(0, min(y, target_size - 1))
                w = max(0, min(w, target_size - x))
                h = max(0, min(h, target_size - y))
                
                # 4. 计算缩放比例
                scale_x = target_size / crop_w
                scale_y = target_size / crop_h
                
                # 5. 应用缩放
                x = x * scale_x
                y = y * scale_y
                w = w * scale_x
                h = h * scale_y
                
                new_ann["bbox"] = [x, y, w, h]
                
                # 更新面积
                if 'area' in new_ann:
                    new_ann["area"] = w * h
            
            # 调整分割标注
            if 'segmentation' in new_ann and isinstance(new_ann['segmentation'], list):
                new_segmentation = []
                for polygon in new_ann['segmentation']:
                    new_polygon = []
                    for i in range(0, len(polygon), 2):
                        px = polygon[i]
                        py = polygon[i+1]
                        
                        # 应用水平翻转
                        if do_flip:
                            px = orig_w - px
                        
                        # 应用裁剪偏移
                        px = px - crop_x
                        py = py - crop_y
                        
                        # 应用缩放
                        px = px * scale_x
                        py = py * scale_y
                        
                        new_polygon.extend([px, py])
                    new_segmentation.append(new_polygon)
                new_ann["segmentation"] = new_segmentation
            
            new_annotations["annotations"].append(new_ann)
            new_ann_id += 1
        
        new_image_id += 1
    
    # 保存新标注文件
    annotation_path = f"{output_dir}/annotations_augmented.json"
    with open(annotation_path, 'w') as f:
        json.dump(new_annotations, f, indent=2)
    

    return new_annotations

if __name__ == "__main__":
    # 配置路径
    input_image_dir = "./coco2024/val2017_resized"
    input_annotation_file = "./coco2024/annotations/instances_val2017_resized.json"
    output_dir = "./coco2024/augmented_dataset"
    
    # 创建增强数据集（处理前20张）
    create_augmented_dataset(
        image_dir=input_image_dir,
        annotation_file=input_annotation_file,
        output_dir=output_dir,
        num_samples=20
    )