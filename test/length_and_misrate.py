import matplotlib.font_manager
from LLMsteg import encode, decode, load_model
import random
import matplotlib.pyplot as plt
import matplotlib
from dotenv import load_dotenv
import os

BYTE_SIZE = 200
BITS_PER_NUMBER = 4

load_dotenv()

def calculate_error_rate(original, received, k):
    original_bits = ''.join(format(x, f'0{k}b') for x in original)
    received_bits = ''.join(format(x, f'0{k}b') for x in received)
    errors = sum(ob != rb for ob, rb in zip(original_bits, received_bits))
    return errors / len(original_bits)

def main():
    model_names = ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-6B-Chat"]
    for model_name in model_names:
        model, tokenizer = load_model(model_name, load_in_4bit=True)

        random_numbers = [random.randint(0, 2**BITS_PER_NUMBER-1) for _ in range(BYTE_SIZE * 8 // BITS_PER_NUMBER)]
        prompt = "Give me a short introduction to large language model."
        response = encode(random_numbers, prompt, model, tokenizer)
        recv = decode(response, prompt, model, tokenizer)

        segment_size = 8//BITS_PER_NUMBER
        error_rates = []
        lengths = []

        for i in range(segment_size, len(random_numbers) + 1, segment_size):
            segment_original = random_numbers[:i]
            segment_received = recv[:i]
            error_rate = calculate_error_rate(segment_original, segment_received, BITS_PER_NUMBER)
            error_rates.append(error_rate)
            lengths.append(i//segment_size)

        # 创建新图形
        plt.figure()
        zhfont=matplotlib.font_manager.FontProperties(fname="LLMsteg/SourceHanSansCN-Regular.otf")
        plt.plot(lengths, error_rates, marker='o')
        plt.xlabel('数据长度（字节）', fontproperties=zhfont)
        plt.ylabel('误码率', fontproperties=zhfont)
        plt.title(f'误码率与数据长度 - {model_name}', fontproperties=zhfont)
        plt.grid(True)
        safe_model_name = model_name.replace("/", "_")
        image_name = f"cache/length_and_misrate_{safe_model_name}.png"
        plt.savefig(image_name, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭当前图形

if __name__ == "__main__":
    main()