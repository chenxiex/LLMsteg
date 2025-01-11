import matplotlib.font_manager
from main import encode, decode
import random
import matplotlib.pyplot as plt
import matplotlib
from modelscope import AutoModelForCausalLM, AutoTokenizer
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
    model_name = os.getenv("MODEL_DIR", "Qwen/Qwen2.5-3B-Instruct")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    random_numbers = [random.randint(0, 2**BITS_PER_NUMBER-1) for _ in range(BYTE_SIZE * 2)]
    prompt = "Give me a short introduction to large language model."
    response = encode(random_numbers, BITS_PER_NUMBER, prompt, model, tokenizer)
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

    zhfont=matplotlib.font_manager.FontProperties(fname="SourceHanSansCN-Regular.otf")
    plt.plot(lengths, error_rates, marker='o')
    plt.xlabel('数据长度（字节）', fontproperties=zhfont)
    plt.ylabel('误码率', fontproperties=zhfont)
    plt.title('误码率与数据长度', fontproperties=zhfont)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()