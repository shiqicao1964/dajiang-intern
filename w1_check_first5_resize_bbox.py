# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 12:11:26 2026

@author: shiqi
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

def visualize_bbox_resize():
    """Visualize bbox resize results for first 5 images"""
    
    # Paths
    original_image_dir = "./coco2024/val2017"
    resized_image_dir = "./coco2024/val2017_resized"
    original_ann_file = "./coco2024/annotations/instances_val2017.json"
    resized_ann_file = "./coco2024/annotations/instances_val2017_resized.json"
    output_dir = "./coco2024/resize_bbox_first5"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    with open(original_ann_file, 'r') as f:
        original_data = json.load(f)
    
    with open(resized_ann_file, 'r') as f:
        resized_data = json.load(f)
    
    # Get first 5 images
    first_5_images = original_data['images'][:5]
    
    for img_info in tqdm(first_5_images, desc="Visualizing bbox resize"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Find annotations for this image
        original_anns = [ann for ann in original_data['annotations'] if ann['image_id'] == img_id]
        resized_anns = [ann for ann in resized_data['annotations'] if ann['image_id'] == img_id]
        
        if not original_anns:
            continue
        
        # Load images
        original_img_path = os.path.join(original_image_dir, file_name)
        resized_img_path = os.path.join(resized_image_dir, file_name)
        
        original_img = Image.open(original_img_path)
        resized_img = Image.open(resized_img_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image with bbox
        axes[0].imshow(original_img)
        for ann in original_anns:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[0].add_patch(rect)
        axes[0].set_title(f'Original: {original_img.size}')
        axes[0].axis('off')
        
        # Resized image with bbox
        axes[1].imshow(resized_img)
        for ann in resized_anns:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[1].add_patch(rect)
        axes[1].set_title(f'Resized: {resized_img.size}')
        axes[1].axis('off')
        
        # Add bbox coordinates info
        bbox_info = ""
        if original_anns and 'bbox' in original_anns[0]:
            orig_bbox = original_anns[0]['bbox']
            resize_bbox = resized_anns[0]['bbox'] if resized_anns else []
            bbox_info = f"\nOriginal bbox: {orig_bbox}\nResized bbox: {resize_bbox}"
        
        plt.suptitle(f'Image ID: {img_id} - {file_name}{bbox_info}', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{img_id}_{file_name.replace('.jpg', '.png')}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    visualize_bbox_resize()