import os
import json
import sys
sys.path.insert(0, sys.path[0]+"/../")
from main import encode
import random
from tqdm import tqdm

def generate():
    # 检查缓存文件是否存在
    cache_file = "./cache/twitter_nlp_generated.json"
    test = []

    if not os.path.exists("./cache"):
        os.makedirs("./cache")
    if os.path.exists(cache_file):
        # 如果缓存文件存在，直接从文件加载
        print(f"加载缓存文件 {cache_file}...")
        with open(cache_file, 'r') as f:
            test = json.load(f)  # json.load可以直接将JSON数组还原为Python列表
    else:
        # 如果缓存文件不存在，执行原来的处理逻辑
        from datasets import load_dataset
        ds = load_dataset("startificial/twitter-nlp")
        
        print(f"处理数据并创建缓存文件 {cache_file}...")
        for text in tqdm(ds["test"]["text"][0:1000], desc="Encoding texts"):
            # 生成随机数数组，长度为33，范围在0-7之间，相当于99个比特位
            a = [random.randint(0, 7) for _ in range(33)] 
            test.append(encode(a, text))
            
        # 将结果保存到缓存文件
        with open(cache_file, 'w') as f:
            json.dump(test, f)  # json.dump可以直接序列化Python列表为JSON数组
    return test
