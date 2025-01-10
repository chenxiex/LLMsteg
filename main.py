from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

def encode(a, k, prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = []
    input_ids = model_inputs.input_ids
    end_tokens = {tokenizer.convert_tokens_to_ids(token) for token in [".", "?", "!"]}

    with torch.no_grad():
        for i in range(len(a)):
            outputs = model(input_ids=input_ids, temperature=0)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            next_token_id = sorted_indices[0, a[i]].unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token_id.item())

        while generated_ids[-1] not in end_tokens:
            outputs = model(input_ids=input_ids, temperature=0)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            next_token_id = sorted_indices[0, 0].unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token_id.item())

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def decode(response, prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    tokens = tokenizer(response, return_tensors="pt").input_ids[0].to(model.device)
    a1 = []
    input_ids = model_inputs.input_ids
    with torch.no_grad():
        for token in tokens:
            outputs = model(input_ids=input_ids, temperature=0)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            token_index = (sorted_indices == token).nonzero(as_tuple=True)[1].item()
            a1.append(token_index)
            input_ids = torch.cat([input_ids, token.view(1, 1)], dim=-1)

    return a1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode_decode', type=int, choices=[0, 1], required=True, help='0 for encode, 1 for decode')
    parser.add_argument('--k', type=int, default=2, help='Value of k')
    parser.add_argument('--prompt', type=str, required=True, help='File path to read prompt')
    parser.add_argument('--secret', type=str, help='File path to read secret data, used when --encode_decode=0')
    parser.add_argument('--cover', type=str, help='File path to read response, used when --encode_decode=1')
    parser.add_argument('--output', type=str, default="output.txt", required=True, help='File path to write output')
    args = parser.parse_args()

    model_name = os.getenv("MODEL_DIR", "./model-dir")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(args.prompt, 'r', errors='ignore') as f:
        prompt = f.read().strip()

    if args.encode_decode == 0:
        a = []
        if args.secret:
            with open(args.secret, 'rb') as f:
                secret_data = f.read()
                bits = ''.join(format(byte, '08b') for byte in secret_data)
                if len(bits) % args.k != 0:
                    bits += '0' * (args.k - len(bits) % args.k)
                for i in range(0, len(bits), args.k):
                    a.append(int(bits[i:i+args.k], 2))
        response = encode(a, args.k, prompt, model, tokenizer)
        with open(args.output, 'w', errors='ignore') as f:
            f.write(response)
    elif args.encode_decode == 1:
        with open(args.cover, 'r', errors='ignore') as f:
            response = f.read().strip()
        a1 = decode(response, prompt, model, tokenizer)
        bits = ''.join(format(x, f'0{args.k}b') for x in a1)
            # 确保bits的长度是8的倍数，不足部分用0填充
        padded_bits = bits.ljust((len(bits) + 7) // 8 * 8, '0')

        # 将二进制字符串按字节分组并转换为字节
        byte_array = bytearray(int(padded_bits[i:i+8], 2) for i in range(0, len(padded_bits), 8))

        # 将字节写入文件
        with open(args.output, 'wb') as binary_file:
            binary_file.write(byte_array)

if __name__ == "__main__":
    main()