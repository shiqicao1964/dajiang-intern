"""
visualize_with_bbox.py
从CSV文件可视化人车图片（带边界框）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from pycocotools.coco import COCO

# 1. 加载数据
csv_file = 'coco_person_car_val.csv'
annotation_file = 'coco/annotations/instances_val2017.json'

df = pd.read_csv(csv_file)
coco = COCO(annotation_file)

# 2. 选择图片
person_samples = df[df['label'] == 0].head(3)
car_samples = df[df['label'] == 1].head(3)


# 创建图形（3x2）
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 4. 下载和显示函数
def download_image(url):
    response = requests.get(url)
    return np.array(Image.open(BytesIO(response.content)).convert('RGB'))

# 显示人的图片（只绘制人的边界框）
for i, (_, sample) in enumerate(person_samples.iterrows()):
    img_id = sample['image_id']
    
    # 下载图片
    img = download_image(sample['coco_url'])
    axes[0, i].imshow(img)
    
    # 获取人的标注
    person_id = coco.getCatIds(['person'])[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_id])
    anns = coco.loadAnns(ann_ids)
    
    # 绘制边界框
    for ann in anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, i].add_patch(rect)
    
    axes[0, i].set_title(f"Person Only\nID: {img_id}\nPersons: {len(anns)}")
    axes[0, i].axis('off')

# 显示车的图片（只绘制车的边界框）
for i, (_, sample) in enumerate(car_samples.iterrows()):
    img_id = sample['image_id']
    
    # 下载图片
    img = download_image(sample['coco_url'])
    axes[1, i].imshow(img)
    
    # 获取车的标注
    car_id = coco.getCatIds(['car'])[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[car_id])
    anns = coco.loadAnns(ann_ids)
    
    # 绘制边界框
    for ann in anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        axes[1, i].add_patch(rect)
    
    axes[1, i].set_title(f"Car Only\nID: {img_id}\nCars: {len(anns)}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()