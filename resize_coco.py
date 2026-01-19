import os
from PIL import Image
import json
from tqdm import tqdm

def resize_images(image_dir, output_dir, size=(224, 224)):
    """Resize images to specified size"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files, desc="Resizing images"):
        img_path = os.path.join(image_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        with Image.open(img_path) as img:
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(output_path)

def update_annotations(original_ann_file, image_dir, output_ann_file, size=(224, 224)):
    """Update bbox, segmentation annotations and image dimensions for resized images"""
    with open(original_ann_file, 'r') as f:
        data = json.load(f)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_set = set(image_files)
    
    # 提取原始尺寸
    original_sizes = {}
    for img_info in data['images']:
        if img_info['file_name'] in image_set:
            img_path = os.path.join(image_dir, img_info['file_name'])
            with Image.open(img_path) as img:
                original_sizes[img_info['id']] = img.size
    
    for img_info in tqdm(data['images'], desc="Updating image dimensions"):
        if img_info['id'] in original_sizes:
            img_info['width'], img_info['height'] = size
    
    for ann in tqdm(data['annotations'], desc="Scaling annotations"):
        if ann['image_id'] in original_sizes:
            original_width, original_height = original_sizes[ann['image_id']]
            x_scale = size[0] / original_width
            y_scale = size[1] / original_height
            
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [x * x_scale, y * y_scale, w * x_scale, h * y_scale]
            
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                new_segmentation = []
                for polygon in ann['segmentation']:
                    new_polygon = []
                    for i in range(0, len(polygon), 2):
                        new_polygon.extend([
                            polygon[i] * x_scale,
                            polygon[i+1] * y_scale
                        ])
                    new_segmentation.append(new_polygon)
                ann['segmentation'] = new_segmentation
    
    with open(output_ann_file, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    image_dir = "./coco2024/val2017"
    output_dir = "./coco2024/val2017_resized"
    original_ann_file = "./coco2024/annotations/instances_val2017.json"
    output_ann_file = "./coco2024/annotations/instances_val2017_resized.json"
    
    resize_images(image_dir, output_dir, size=(224, 224))
    update_annotations(original_ann_file, image_dir, output_ann_file, size=(224, 224))