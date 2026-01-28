import os
import json
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ImageNet标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def normalize_with_imagenet(img_array):
    """ImageNet标准化"""
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    normalized = np.zeros_like(img_array)
    for i in range(3):
        normalized[:, :, i] = (img_array[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    return normalized

def apply_scaling_to_annotations(ann, scale_w, scale_h, do_flip=False, target_w=224, target_h=224, crop_offset=(0, 0), crop_scale=1.0):
    """应用缩放、翻转、裁剪到标注"""
    new_ann = ann.copy()
    
    # 处理bbox
    if 'bbox' in new_ann:
        x, y, w, h = new_ann['bbox']
        
        # 1. 缩放
        x *= scale_w
        y *= scale_h
        w *= scale_w
        h *= scale_h
        
        # 2. 水平翻转
        if do_flip:
            x = target_w - x - w
        
        # 3. 裁剪偏移
        x -= crop_offset[0]
        y -= crop_offset[1]
        
        # 4. 裁剪后resize的缩放
        x *= crop_scale
        y *= crop_scale
        w *= crop_scale
        h *= crop_scale
        
        # 边界检查
        x = max(0, min(x, target_w))
        y = max(0, min(y, target_h))
        w = min(w, target_w - x)
        h = min(h, target_h - y)
        
        if w > 2 and h > 2:  # 过滤太小的标注
            new_ann["bbox"] = [x, y, w, h]
            new_ann["area"] = w * h
        else:
            return None  # 标注太小，返回None
   
    
    # 处理segmentation
    if 'segmentation' in new_ann and isinstance(new_ann['segmentation'], list):
        new_segmentation = []
        for polygon in new_ann['segmentation']:
            if isinstance(polygon, list):
                new_polygon = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i]
                    py = polygon[i+1]
                    
                    # 1. 缩放
                    px *= scale_w
                    py *= scale_h
                    
                    # 2. 水平翻转
                    if do_flip:
                        px = target_w - px
                    
                    # 3. 裁剪偏移
                    px -= crop_offset[0]
                    py -= crop_offset[1]
                    
                    # 4. 裁剪后resize的缩放
                    px *= crop_scale
                    py *= crop_scale
                    
                    # 5. 边界检查 - 确保在图像范围内
                    px = max(0, min(px, target_w))
                    py = max(0, min(py, target_h))
                    
                    new_polygon.extend([px, py])
                
                # 过滤掉无效的多边形（点数太少或面积太小）
                if len(new_polygon) >= 6:  # 至少需要3个点（6个坐标值）
                    # 检查多边形是否有效（面积不为0）
                    x_coords = new_polygon[0::2]
                    y_coords = new_polygon[1::2]
                    if len(set(x_coords)) > 1 or len(set(y_coords)) > 1:  # 不是所有点都在同一位置
                        new_segmentation.append(new_polygon)
        
        if new_segmentation:
            new_ann["segmentation"] = new_segmentation
        else:
            # 如果没有有效的分割标注，返回None
            if 'bbox' not in new_ann:
                return None
            
    
    # # 处理segmentation
    # if 'segmentation' in new_ann and isinstance(new_ann['segmentation'], list):
    #     new_segmentation = []
    #     for polygon in new_ann['segmentation']:
    #         if isinstance(polygon, list):
    #             new_polygon = []
    #             for i in range(0, len(polygon), 2):
    #                 px = polygon[i]
    #                 py = polygon[i+1]
                    
    #                 # 1. 缩放
    #                 px *= scale_w
    #                 py *= scale_h
                    
    #                 # 2. 水平翻转
    #                 if do_flip:
    #                     px = target_w - px
                    
    #                 # 3. 裁剪偏移
    #                 px -= crop_offset[0]
    #                 py -= crop_offset[1]
                    
    #                 # 4. 裁剪后resize的缩放
    #                 px *= crop_scale
    #                 py *= crop_scale
                    
    #                 new_polygon.extend([px, py])
    #             if len(new_polygon) > 0:
    #                 new_segmentation.append(new_polygon)
    #     if new_segmentation:
    #         new_ann["segmentation"] = new_segmentation
    #     else:
    #         # 如果没有有效的分割标注，返回None
    #         if 'bbox' not in new_ann:
    #             return None
    
    return new_ann

def process_and_augment_dataset(
    image_dir="./coco2024/val2017",
    ann_file="./coco2024/annotations/instances_val2017.json",
    output_dir="./coco2024/processed",
    target_size=224,
    num_samples=100,
    seed=42
):
    """处理图像：1.缩放保存 -> 2.增强保存 -> 标准化.pt文件"""
    
    random.seed(seed)
    
    # 创建输出目录
    resize_output_dir = f"{output_dir}/resized_images"  # 缩放后的图像
    aug_output_dir = f"{output_dir}/augmented_images"   # 增强后的图像
    pt_output_dir = f"{output_dir}/pt_files"           # 标准化.pt文件
    os.makedirs(resize_output_dir, exist_ok=True)
    os.makedirs(aug_output_dir, exist_ok=True)
    os.makedirs(pt_output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    
    # 1. 加载原始标注
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # 2. 创建图像ID到信息的映射
    img_info_dict = {img['id']: img for img in data['images']}
    img_anns_dict = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns_dict:
            img_anns_dict[img_id] = []
        img_anns_dict[img_id].append(ann)
    
    # 3. 选择指定数量的图像
    all_img_ids = list(img_info_dict.keys())
    selected_ids = all_img_ids[:min(num_samples, len(all_img_ids))]
    
    print(f"处理 {len(selected_ids)} 张图像，将生成 {len(selected_ids)*2} 张图像...")
    
    # 4. 新标注数据结构
    new_annotations = {
        "images": [],
        "annotations": [],
        "categories": data['categories']
    }
    
    new_img_id = 1
    new_ann_id = 1
    
    # 5. 处理每张图像
    for img_id in tqdm(selected_ids, desc="Processing"):
        if img_id not in img_info_dict:
            continue
            
        img_info = img_info_dict[img_id]
        img_path = f"{image_dir}/{img_info['file_name']}"
        
        if not os.path.exists(img_path):
            continue
        
        # 5.1 加载原始图像
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # 计算缩放比例
        scale_w = target_size / orig_w
        scale_h = target_size / orig_h
        
        # ========== 第一阶段：只做Resize ==========
        # 5.2 Resize图像
        resized_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 5.3 保存resize后的图像
        resize_filename = f"resize_{img_info['file_name']}"
        resized_img.save(f"{resize_output_dir}/{resize_filename}")
        
        # 5.4 转换为numpy并标准化，保存为.pt
        img_array = np.array(resized_img).astype(np.float32)
        normalized_array = normalize_with_imagenet(img_array)
        pt_filename = f"resize_{os.path.splitext(img_info['file_name'])[0]}.pt"
        torch.save(torch.from_numpy(normalized_array).permute(2, 0, 1), f"{pt_output_dir}/{pt_filename}")
        
        # 5.5 添加resize图像信息
        resize_img_id = new_img_id
        new_annotations["images"].append({
            "id": resize_img_id,
            "width": target_size,
            "height": target_size,
            "file_name": resize_filename,
            "pt_file": pt_filename,
            "original_id": img_id,
            "type": "resized"
        })
        
        # 5.6 处理resize图像的标注
        if img_id in img_anns_dict:
            for ann in img_anns_dict[img_id]:
                new_ann = apply_scaling_to_annotations(
                    ann, 
                    scale_w=scale_w, 
                    scale_h=scale_h, 
                    do_flip=False,
                    target_w=target_size,
                    target_h=target_size,
                    crop_offset=(0, 0),
                    crop_scale=1.0
                )
                if new_ann is not None:  # 只添加有效的标注
                    new_ann["id"] = new_ann_id
                    new_ann["image_id"] = resize_img_id
                    new_annotations["annotations"].append(new_ann)
                    new_ann_id += 1
        
        new_img_id += 1
        
        # ========== 第二阶段：数据增强 ==========
        # 5.7 随机增强：水平翻转
        do_flip = random.random() < 0.5
        aug_img = resized_img.copy()
        if do_flip:
            aug_img = ImageOps.mirror(aug_img)
        
        # 5.8 随机裁剪 裁剪保留一半以上
        crop_size = random.randint(target_size // 2, target_size)
        crop_x = random.randint(0, target_size - crop_size) if target_size > crop_size else 0
        crop_y = random.randint(0, target_size - crop_size) if target_size > crop_size else 0
        
        # 计算裁剪后的缩放比例
        crop_scale = target_size / crop_size
        
        # 执行裁剪
        cropped_img = aug_img.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        final_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 5.9 保存增强后的图像
        aug_filename = f"aug_{img_info['file_name']}"
        final_img.save(f"{aug_output_dir}/{aug_filename}")
        
        # 5.10 转换为numpy并标准化，保存为.pt
        img_array_aug = np.array(final_img).astype(np.float32)
        normalized_array_aug = normalize_with_imagenet(img_array_aug)
        pt_filename_aug = f"aug_{os.path.splitext(img_info['file_name'])[0]}.pt"
        torch.save(torch.from_numpy(normalized_array_aug).permute(2, 0, 1), f"{pt_output_dir}/{pt_filename_aug}")
        
        # 5.11 添加增强图像信息
        aug_img_id = new_img_id
        new_annotations["images"].append({
            "id": aug_img_id,
            "width": target_size,
            "height": target_size,
            "file_name": aug_filename,
            "pt_file": pt_filename_aug,
            "original_id": img_id,
            "type": "augmented",
            "augmentation": {
                "flipped": do_flip,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_size": crop_size,
                "crop_scale": crop_scale
            }
        })
        
        # 5.12 处理增强图像的标注
        if img_id in img_anns_dict:
            for ann in img_anns_dict[img_id]:
                new_ann = apply_scaling_to_annotations(
                    ann, 
                    scale_w=scale_w, 
                    scale_h=scale_h, 
                    do_flip=do_flip,
                    target_w=target_size,
                    target_h=target_size,
                    crop_offset=(crop_x, crop_y),
                    crop_scale=crop_scale
                )
                if new_ann is not None:  # 只添加有效的标注
                    new_ann["id"] = new_ann_id
                    new_ann["image_id"] = aug_img_id
                    new_annotations["annotations"].append(new_ann)
                    new_ann_id += 1
        
        new_img_id += 1
    
    # 6. 保存新标注文件
    with open(f"{output_dir}/annotations/processed.json", 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"Resized图像: {resize_output_dir} ({len(selected_ids)}张)")
    print(f"Augmented图像: {aug_output_dir} ({len(selected_ids)}张)")
    print(f".pt文件: {pt_output_dir} ({len(selected_ids)*2}个)")
    print(f"标注文件: {output_dir}/annotations/processed.json")
    print(f"总图像数: {len(new_annotations['images'])}张")
    print(f" 总标注数: {len(new_annotations['annotations'])}个")
    
    return new_annotations

def load_pt_image(pt_path):
    """加载.pt文件"""
    tensor = torch.load(pt_path)
    return tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC

if __name__ == "__main__":
    # 配置参数
    image_dir = "./coco2024/val2017"
    ann_file = "./coco2024/annotations/instances_val2017.json"
    output_dir = "./coco2024/processed"
    
    # 运行处理流程
    annotations = process_and_augment_dataset(
        image_dir=image_dir,
        ann_file=ann_file,
        output_dir=output_dir,
        target_size=224,
        num_samples=100,
        seed=42
    )