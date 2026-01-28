import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils

def visualize_segmentation_resize():
    """Visualize segmentation resize results for first 5 images"""
    
    # Paths
    original_image_dir = "./coco2024/val2017"
    resized_image_dir = "./coco2024/val2017_resized"
    original_ann_file = "./coco2024/annotations/instances_val2017.json"
    resized_ann_file = "./coco2024/annotations/instances_val2017_resized.json"
    output_dir = "./coco2024/resize_seg_first5"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    with open(original_ann_file, 'r') as f:
        original_data = json.load(f)
    
    with open(resized_ann_file, 'r') as f:
        resized_data = json.load(f)
    
    # Get first 5 images
    first_5_images = original_data['images'][:5]
    
    # 使用高对比度颜色
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for img_info in tqdm(first_5_images, desc="Visualizing segmentation resize"):
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Original image with segmentation (高对比度颜色)
        axes[0, 0].imshow(original_img)
        for idx, ann in enumerate(original_anns):
            color = colors[idx % len(colors)]
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for polygon in ann['segmentation']:
                    polygon_array = np.array(polygon).reshape(-1, 2)
                    axes[0, 0].plot(polygon_array[:, 0], polygon_array[:, 1], color=color, 
                                   linewidth=3, alpha=0.8, linestyle='-', marker='o', markersize=3)
        axes[0, 0].set_title(f'Original Segmentation\nSize: {original_img.size}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Resized image with segmentation (高对比度颜色)
        axes[0, 1].imshow(resized_img)
        for idx, ann in enumerate(resized_anns):
            color = colors[idx % len(colors)]
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for polygon in ann['segmentation']:
                    polygon_array = np.array(polygon).reshape(-1, 2)
                    axes[0, 1].plot(polygon_array[:, 0], polygon_array[:, 1], color=color,
                                   linewidth=3, alpha=0.8, linestyle='-', marker='o', markersize=3)
        axes[0, 1].set_title(f'Resized Segmentation\nSize: {resized_img.size}', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Original image with filled segmentation mask (使用viridis等高对比度colormap)
        axes[1, 0].imshow(original_img)
        mask_overlay = np.zeros((img_info['height'], img_info['width'], 4), dtype=np.float32)
        
        for idx, ann in enumerate(original_anns):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list) and ann['segmentation']:
                try:
                    rles = maskUtils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    rle = maskUtils.merge(rles)
                    mask = maskUtils.decode(rle)
                    
                    # 为每个mask分配不同颜色
                    color_idx = idx % len(colors)
                    rgba_color = plt.cm.tab10(color_idx / len(colors))  # 使用tab10 colormap
                    
                    # 创建带透明度的mask
                    for c in range(3):
                        mask_overlay[..., c] += mask * rgba_color[c]
                    mask_overlay[..., 3] += mask * 0.6  # 透明度
                    
                except:
                    pass
        
        # 限制颜色值在0-1之间
        mask_overlay[..., :3] = np.clip(mask_overlay[..., :3], 0, 1)
        mask_overlay[..., 3] = np.clip(mask_overlay[..., 3], 0, 1)
        
        axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title(f'Original Segmentation Masks\n(High Contrast Colors)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Resized image with filled segmentation mask (使用plasma等高对比度colormap)
        axes[1, 1].imshow(resized_img)
        resized_mask_overlay = np.zeros((224, 224, 4), dtype=np.float32)
        
        for idx, ann in enumerate(resized_anns):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list) and ann['segmentation']:
                try:
                    rles = maskUtils.frPyObjects(ann['segmentation'], 224, 224)
                    rle = maskUtils.merge(rles)
                    mask = maskUtils.decode(rle)
                    
                    # 为每个mask分配不同颜色，使用与原始对应的颜色
                    color_idx = idx % len(colors)
                    rgba_color = plt.cm.tab20(color_idx / len(colors))  # 使用tab20 colormap，对比度更高
                    
                    # 创建带透明度的mask
                    for c in range(3):
                        resized_mask_overlay[..., c] += mask * rgba_color[c]
                    resized_mask_overlay[..., 3] += mask * 0.7  # 更高透明度
                    
                except:
                    pass
        
        # 限制颜色值在0-1之间
        resized_mask_overlay[..., :3] = np.clip(resized_mask_overlay[..., :3], 0, 1)
        resized_mask_overlay[..., 3] = np.clip(resized_mask_overlay[..., 3], 0, 1)
        
        axes[1, 1].imshow(resized_mask_overlay)
        axes[1, 1].set_title(f'Resized Segmentation Masks\n(High Contrast Colors)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 添加图例
        if len(original_anns) > 0:
            legend_elements = []
            for idx in range(min(len(original_anns), 6)):
                color = colors[idx % len(colors)]
                legend_elements.append(patches.Patch(facecolor=color, edgecolor='black', 
                                                   alpha=0.6, label=f'Object {idx+1}'))
            
            fig.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=10)
        
        plt.suptitle(f'Image ID: {img_id} - {file_name}\nSegmentation Resize Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为图例留出空间
        
        # Save figure
        output_path = os.path.join(output_dir, f"{img_id}_{file_name.replace('.jpg', '.png')}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"High contrast visualizations saved to: {output_dir}")

if __name__ == "__main__":
    visualize_segmentation_resize()