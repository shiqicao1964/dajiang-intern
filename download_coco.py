# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:06:38 2026

@author: shiqi
"""

import os
import requests
import zipfile
from tqdm import tqdm

class COCO2024Downloader:
    def __init__(self, download_dir="./coco2024"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # 最新的 COCO 下载链接（2024年）
        self.latest_urls = {
            # 基础数据集（推荐从这里开始）
            "coco_2017": {
                "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                "test_images": "http://images.cocodataset.org/zips/test2017.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            },
            
            # 2024年新增/更新（如果有的话）
            "coco_2024_updates": {
                # 注意：COCO 2024 可能还没有完全发布
                # 通常更新会在这里：http://images.cocodataset.org/annotations/
                "panoptic": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
                "stuff": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
                "densepose": "https://github.com/facebookresearch/DensePose/raw/master/DENSEPOSE_IUV.zip"
            }
        }
    
    def download_with_progress(self, url, save_path):
        """带进度条的下载函数"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=f"下载 {os.path.basename(save_path)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    def download_coco_2017(self, components=None):
        """下载 COCO 2017 数据集（目前最新完整版）"""
        if components is None:
            components = ["train_images", "val_images", "annotations"]
        
        print("开始下载 COCO 2017 数据集...")
        print("=" * 60)
        
        for component in components:
            if component in self.latest_urls["coco_2017"]:
                url = self.latest_urls["coco_2017"][component]
                filename = os.path.join(self.download_dir, os.path.basename(url))
                
                print(f"\n📥 下载 {component}...")
                print(f"   URL: {url}")
                
                try:
                    self.download_with_progress(url, filename)
                    
                    # 解压文件
                    print(f"   📦 解压文件...")
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(self.download_dir)
                    
                    print(f"   ✅ {component} 下载完成！")
                    
                    # 可选：删除压缩包节省空间
                    # os.remove(filename)
                    
                except Exception as e:
                    print(f"   ❌ 下载失败: {e}")
        
        print("\n" + "=" * 60)
        print("✅ COCO 2017 数据集下载完成！")
        print(f"   位置: {os.path.abspath(self.download_dir)}")
    
    def check_latest_updates(self):
        """检查是否有更新的版本"""
        print("检查 COCO 数据集最新更新...")
        print("=" * 60)
        
        # 可以在这里添加检查最新版本的逻辑
        print("当前最新完整版本: COCO 2017")
        print("COCO 2024 仍在开发中，尚未完全发布")
        print("建议使用 COCO 2017 进行学习和研究")
        
        return "coco_2017"

# 使用示例
if __name__ == "__main__":
    downloader = COCO2024Downloader()
    
    # 检查最新版本
    latest_version = downloader.check_latest_updates()
    
    # 下载 COCO 2017（目前最新完整版）
    downloader.download_coco_2017([
        "val_images",        # 验证集图像（1GB，适合测试）
        "annotations"        # 标注文件
        # "train_images",    # 训练集图像（18GB，需要时再下载）
        # "test_images"      # 测试集图像（6GB）
    ])