"""
download_coco_async.py
异步高速下载COCO人车分类图片
"""
"""
download_coco_fixed.py
修复Windows上事件循环问题的下载脚本
"""

import asyncio
import aiohttp
import pandas as pd
import os
import sys
from tqdm import tqdm
import time

class AsyncCOCODownloader:
    def __init__(self, max_concurrent=100, timeout=30):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
    async def download_image(self, session, url, save_path, semaphore):
        """异步下载单个图片"""
        async with semaphore:
            if os.path.exists(save_path):
                return {'status': 'skipped', 'size': 0, 'file': os.path.basename(save_path)}
            
            try:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(save_path, 'wb') as f:
                            f.write(content)
                        return {
                            'status': 'success',
                            'size': len(content),
                            'file': os.path.basename(save_path)
                        }
                    else:
                        return {
                            'status': f'failed_http_{response.status}',
                            'size': 0,
                            'file': os.path.basename(save_path)
                        }
            except asyncio.TimeoutError:
                return {'status': 'failed_timeout', 'size': 0, 'file': os.path.basename(save_path)}
            except Exception as e:
                return {'status': f'failed_{str(e)[:30]}', 'size': 0, 'file': os.path.basename(save_path)}
    
    async def download_batch(self, df, output_dir, desc="下载进度"):
        """异步下载一批图片"""
        os.makedirs(output_dir, exist_ok=True)
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for _, row in df.iterrows():
                save_path = os.path.join(output_dir, row['file_name'])
                task = self.download_image(session, row['coco_url'], save_path, semaphore)
                tasks.append(task)
            
            results = []
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
            
            return results
    
    def download_dataset(self, csv_file, output_dir, dataset_name="数据集"):
        """下载整个数据集"""
        print(f"\n{'='*60}")
        print(f"开始下载 {dataset_name}")
        print(f"{'='*60}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"📁 文件: {csv_file}")
            print(f"📊 图片数量: {len(df):,}张")
            print(f"📂 输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 加载CSV失败: {e}")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 修复：Windows上正确处理事件循环
        start_time = time.time()
        
        # 方法1：尝试获取现有循环，如果不存在则创建新循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 检查循环是否在运行
        if loop.is_running():
            # 如果循环已经在运行，使用不同的方法
            print("⚠️  事件循环已在运行，使用nest_asyncio解决...")
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                print("正在安装nest_asyncio...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
                import nest_asyncio
                nest_asyncio.apply()
        
        # 运行异步任务
        try:
            results = loop.run_until_complete(
                self.download_batch(df, output_dir, desc=f"{dataset_name}下载进度")
            )
        except RuntimeError as e:
            if "already running" in str(e):
                # 如果还有问题，使用asyncio.run（Python 3.7+）
                print("使用asyncio.run()...")
                results = asyncio.run(self.download_batch(df, output_dir, desc=f"{dataset_name}下载进度"))
            else:
                raise
        
        end_time = time.time()
        
        # 统计结果
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = len(results) - success - skipped
        
        total_time = end_time - start_time
        speed = success / total_time if total_time > 0 else 0
        
        print(f"\n📈 {dataset_name}下载完成!")
        print(f"   ✅ 成功: {success:,}张")
        print(f"   ⏭️  跳过: {skipped:,}张")
        print(f"   ❌ 失败: {failed:,}张")
        print(f"   ⏱️  耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"   🚀 速度: {speed:.1f}张/秒 ({speed*3600:.0f}张/小时)")
        
        if failed > 0:
            print(f"\n⚠️  失败详情（前10个）:")
            failed_items = [r for r in results if r['status'].startswith('failed')]
            for i, r in enumerate(failed_items[:10]):
                print(f"   {i+1}. {r['file']}: {r['status']}")
        
        return {
            'success': success,
            'skipped': skipped,
            'failed': failed,
            'time': total_time,
            'speed': speed
        }

def main():
    """主函数"""
    print("🚀 COCO人车分类图片异步下载工具（Windows修复版）")
    print("="*60)
    
    # 配置参数
    config = {
        'max_concurrent': 100,  # Windows建议不要太高
        'timeout': 30,
    }
    
    # 文件路径配置
    datasets = [
        {
            'name': '训练集',
            'csv_file': 'coco_person_car_train.csv',
            'output_dir': 'coco/images/person_car_train2017',
            'new_csv': 'coco_person_car_train_local.csv'
        },
        {
            'name': '验证集', 
            'csv_file': 'coco_person_car_val.csv',
            'output_dir': 'coco/images/person_car_val2017',
            'new_csv': 'coco_person_car_val_local.csv'
        }
    ]
    
    # 检查CSV文件
    for dataset in datasets:
        if not os.path.exists(dataset['csv_file']):
            print(f"❌ 找不到CSV文件: {dataset['csv_file']}")
            print("请先运行数据集构建脚本生成CSV文件")
            sys.exit(1)
    
    # 创建下载器
    downloader = AsyncCOCODownloader(
        max_concurrent=config['max_concurrent'],
        timeout=config['timeout']
    )
    
    total_stats = {
        'total_images': 0,
        'total_success': 0,
        'total_failed': 0,
        'total_time': 0
    }
    
    # 下载所有数据集
    for dataset in datasets:
        stats = downloader.download_dataset(
            csv_file=dataset['csv_file'],
            output_dir=dataset['output_dir'],
            dataset_name=dataset['name']
        )
        
        if stats:
            # 更新CSV路径
            df = pd.read_csv(dataset['csv_file'])
            df['file_path'] = df['file_name'].apply(
                lambda x: os.path.join(dataset['output_dir'], x)
            )
            df.to_csv(dataset['new_csv'], index=False)
            print(f"\n📄 已更新CSV路径: {dataset['new_csv']}")
            
            # 累计统计
            total_stats['total_success'] += stats['success']
            total_stats['total_failed'] += stats['failed']
            total_stats['total_time'] += stats['time']
            
            df_size = len(pd.read_csv(dataset['csv_file']))
            total_stats['total_images'] += df_size
    
    # 总体统计
    print(f"\n{'='*60}")
    print("🎉 全部下载完成!")
    print(f"{'='*60}")
    print(f"📊 总体统计:")
    print(f"   总图片数: {total_stats['total_images']:,}张")
    print(f"   成功下载: {total_stats['total_success']:,}张")
    print(f"   下载失败: {total_stats['total_failed']:,}张")
    print(f"   总耗时: {total_stats['total_time']/60:.1f}分钟")
    
    print(f"\n📁 目录结构:")
    print(f"   coco/images/person_car_train2017/")
    print(f"   coco/images/person_car_val2017/")
    
    print(f"\n📄 生成的CSV文件:")
    print(f"   coco_person_car_train_local.csv")
    print(f"   coco_person_car_val_local.csv")
    
    print(f"\n✅ 可以开始训练了!")

if __name__ == "__main__":
    # 检查依赖
    try:
        import aiohttp
        import pandas
        from tqdm import tqdm
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install aiohttp pandas tqdm")
        sys.exit(1)
    
    # 运行主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  下载被用户中断")
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        import traceback
        traceback.print_exc()