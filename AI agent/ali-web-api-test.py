import requests
import json

# 1. 您的API地址
api_url = "https://ai-agent-env-cvobsfpbbg.cn-hangzhou.fcapp.run/detect"

# 2. 要发送的数据
data = {
    "max_distance": 2.5
}

# 3. 发送POST请求
try:
    print(f"📤 发送POST请求到: {api_url}")
    print(f"📦 发送数据: {json.dumps(data, ensure_ascii=False)}")
    
    response = requests.post(
        api_url,
        json=data,  # 自动转换为JSON
        headers={'Content-Type': 'application/json'},
        timeout=10
    )
    
    print(f"\n📥 收到响应:")
    print(f"   状态码: {response.status_code}")
    
    # 4. 显示结果
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 请求成功!")
        print(f"\n🌿 检测到 {result.get('count', 0)} 株植物:")
        
        for plant in result.get('detected_plants', []):
            print(f"   - {plant['name']} ({plant['id']})")
            print(f"     位置: x={plant['position']['x']}, y={plant['position']['y']}")
            print(f"     距离: {plant['distance']}米, 健康: {plant['health']}")
    else:
        print(f"❌ 请求失败: {response.text}")
        
except Exception as e:
    print(f"🚨 发生错误: {str(e)}")