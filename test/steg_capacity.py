import LLMsteg
import random

BYTE_SIZE = 64
BITS_PER_NUMBER = 4

def main():
    model_names = ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-6B-Chat"]
    prompt = "Give me a short introduction to large language model."
    for model_name in model_names:
        model, tokenizer = LLMsteg.load_model(model_name=model_name, load_in_4bit=True)
        avg_byte_size = 0
        test_cnt = 3
        for i in range(test_cnt):
            random_numbers = [random.randint(0, 2**BITS_PER_NUMBER-1) for _ in range(BYTE_SIZE * 2)]
            response = LLMsteg.encode(random_numbers, prompt, model, tokenizer)
            # 统计response的字节数
            byte_size = len(response.encode('utf-8'))
            avg_byte_size += byte_size
        avg_byte_size /= test_cnt
        # 计算每个字节的隐写容量
        steg_capacity = 64 / avg_byte_size * 8 / 0.7101 # 0.7101是LLMzip的压缩率
        print(f"Model: {model_name}, Steg Capacity: {steg_capacity:.2f} tokens per byte")

if __name__ == "__main__":
    main()