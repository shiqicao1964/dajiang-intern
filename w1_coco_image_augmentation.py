import os
import random
import json
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from tqdm import tqdm

def create_augmented_dataset(image_dir, annotation_file, output_dir, num_samples=None, seed=42):
    """创建数据增强数据集（随机水平翻转+随机裁剪）"""
    
    random.seed(seed)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()[:num_samples] if num_samples else coco.getImgIds()
    
    new_annotations = {
        "info": coco.dataset.get('info', {}),
        "licenses": coco.dataset.get('licenses', []),
        "images": [],
        "annotations": [],
        "categories": coco.dataset.get('categories', [])
    }
    
    new_image_id = 100000
    new_ann_id = 100000
    target_size = 224
    
    for img_id in tqdm(image_ids, desc="Processing"):
        img_info = coco.loadImgs([img_id])[0]
        image = Image.open(f"{image_dir}/{img_info['file_name']}").convert('RGB')
        orig_w, orig_h = image.size
        
        # 水平翻转
        do_flip = random.random() < 0.5
        if do_flip:
            image = ImageOps.mirror(image)
        
        # 随机裁剪
        if orig_w < target_size or orig_h < target_size:
            # 原图太小，使用原图尺寸
            crop_size = min(orig_w, orig_h)
            crop_x = max(0, (orig_w - crop_size) // 2)
            crop_y = max(0, (orig_h - crop_size) // 2)
        else:
            # 正常裁剪
            crop_size = target_size
            crop_x = random.randint(0, orig_w - crop_size)
            crop_y = random.randint(0, orig_h - crop_size)
        
        # 执行裁剪
        cropped_image = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
        
        # 保存图像（这里保存！）
        new_filename = f"aug_{img_info['file_name']}"
        save_path = f"{output_dir}/images/{new_filename}"
        cropped_image.save(save_path)  # 这里保存图像！
        
        actual_width, actual_height = cropped_image.size
        
        # 添加图像信息
        new_annotations["images"].append({
            "id": new_image_id,
            "width": actual_width,
            "height": actual_height,
            "file_name": new_filename,
            "_original_id": img_id
        })
        
        # 处理标注
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        for ann in coco.loadAnns(ann_ids):
            new_ann = ann.copy()
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = new_image_id
            
            # 调整bbox
            if 'bbox' in new_ann:
                x, y, w, h = new_ann['bbox']
                
                if do_flip:
                    x = orig_w - x - w
                
                x = x - crop_x
                y = y - crop_y
                
                # 过滤裁剪区域外的bbox
                if x + w < 0 or x > crop_size or y + h < 0 or y > crop_size:
                    continue
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, crop_size - x)
                h = min(h, crop_size - y)
                
                new_ann["bbox"] = [x, y, w, h]
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
                        
                        if do_flip:
                            px = orig_w - px
                        
                        px = px - crop_x
                        py = py - crop_y
                        
                        new_polygon.extend([px, py])
                    new_segmentation.append(new_polygon)
                new_ann["segmentation"] = new_segmentation
            
            new_annotations["annotations"].append(new_ann)
            new_ann_id += 1
        
        new_image_id += 1
    
    # 保存标注
    with open(f"{output_dir}/annotations_augmented.json", 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"✅ 完成: {len(new_annotations['images'])} 张图像")
    return new_annotations

if __name__ == "__main__":
    create_augmented_dataset(
        image_dir="./coco2024/val2017",
        annotation_file="./coco2024/annotations/instances_val2017.json",
        output_dir="./coco2024/augmented_dataset",
        num_samples=20
    )