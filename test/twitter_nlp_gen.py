import os
import json
from LLMsteg import encode, load_model
import random
from tqdm import tqdm

BITS_PER_NUMBER = 4

def generate(model_name:str, byte_size=64):
    # 检查缓存文件是否存在
    safe_model_name = model_name.replace("/", "_")
    cache_file = f"cache/twitter_nlp_generated_{safe_model_name}_{byte_size}.json"
    test = []

    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        # 如果缓存文件存在，直接从文件加载
        print(f"加载缓存文件 {cache_file}...")
        with open(cache_file, 'r') as f:
            test = json.load(f)  # json.load可以直接将JSON数组还原为Python列表
    else:
        # 如果缓存文件不存在，执行原来的处理逻辑
        from datasets import load_dataset
        ds = load_dataset("startificial/twitter-nlp")

        model, tokenizer = load_model(model_name)
        
        print(f"处理数据并创建缓存文件 {cache_file}...")
        for text in tqdm(ds["test"]["text"][100:200], desc="Encoding texts"):
            random_numbers = [random.randint(0, 2**BITS_PER_NUMBER-1) for _ in range(byte_size * 8 // BITS_PER_NUMBER)]
            test.append(encode(random_numbers, text, model, tokenizer))

        # 将结果保存到缓存文件
        with open(cache_file, 'w') as f:
            json.dump(test, f)  # json.dump可以直接序列化Python列表为JSON数组
    return test
