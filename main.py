from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
import torch
import argparse
from dotenv import load_dotenv
import os
import re

# 加载.env文件中的环境变量
load_dotenv()

model_name = os.getenv("MODEL_DIR", "Qwen/Qwen3-4B")

system_message = "You are a forum user. The input consists of a post and its replies. Based on the information provided, compose a new reply that contributes meaningfully to the discussion. Your response should be natural, relevant, and written in the tone of an engaged forum participant. Feel free to reference or build upon previous replies where appropriate."

def load_model(model_name=model_name, load_in_4bit=False):
    # 加载模型和分词器
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def encode(a, prompt, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    assert model is not None, "Model is not loaded."
    assert tokenizer is not None, "Tokenizer is not loaded."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = []
    input_ids = model_inputs.input_ids
    end_tokens = {tokenizer.convert_tokens_to_ids(token) for token in [".", "?", "!"]}
    # 添加EOS token到end_tokens集合
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        end_tokens.add(eos_token_id)

    with torch.no_grad():
        for i in range(len(a)):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            next_token_id = sorted_indices[0, a[i]].unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token_id.item())

        while generated_ids[-1] not in end_tokens and len(generated_ids) < 1000:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            next_token_id = sorted_indices[0, 0].unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token_id.item())

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def decode(response, prompt, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    assert model is not None, "Model is not loaded."
    assert tokenizer is not None, "Tokenizer is not loaded."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    tokens = tokenizer(response, return_tensors="pt").input_ids[0].to(model.device)
    a1 = []
    input_ids = model_inputs.input_ids
    with torch.no_grad():
        for token in tokens:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            token_index = (sorted_indices == token).nonzero(as_tuple=True)[1].item()
            a1.append(token_index)
            input_ids = torch.cat([input_ids, token.view(1, 1)], dim=-1)

    return a1

def encode_file(prompt, secret, output, k=4):
    with open(prompt, 'r') as f:
        prompt = f.read().strip()
    a = []
    if secret is not None:
        with open(secret, 'rb') as f:
            secret_data = f.read()
            bits = ''.join(format(byte, '08b') for byte in secret_data)
            if len(bits) % k != 0:
                bits += '0' * (k - len(bits) % k)
            for i in range(0, len(bits), k):
                a.append(int(bits[i:i+k], 2))
    response = encode(a, prompt)
    with open(output, 'w', errors='ignore') as f:
        f.write(response)

def decode_file(prompt, cover, output, k=4):
    with open(prompt, 'r') as f:
        prompt = f.read().strip()
    with open(cover, 'r') as f:
        response = f.read().strip()
    a1 = decode(response, prompt)
    bits = ''.join(format(x, f'0{k}b') for x in a1)
        # 确保bits的长度是8的倍数，不足部分用0填充
    padded_bits = bits.ljust((len(bits) + 7) // 8 * 8, '0')

    # 将二进制字符串按字节分组并转换为字节
    byte_array = bytearray(int(padded_bits[i:i+8], 2) for i in range(0, len(padded_bits), 8))

    # 将字节写入文件
    with open(output, 'wb') as binary_file:
        binary_file.write(byte_array)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode_decode', type=int, choices=[0, 1], required=True, help='0 for encode, 1 for decode')
    parser.add_argument('--k', type=int, default=4, help='Value of k')
    parser.add_argument('--prompt', type=str, required=True, help='File path to read prompt')
    parser.add_argument('--secret', type=str, help='File path to read secret data, used when --encode_decode=0')
    parser.add_argument('--cover', type=str, help='File path to read response, used when --encode_decode=1')
    parser.add_argument('--output', type=str, default="output.txt", required=True, help='File path to write output')
    args = parser.parse_args()

    if args.encode_decode == 0:
        encode_file(args.prompt, args.secret, args.output, args.k)
    elif args.encode_decode == 1:
        decode_file(args.prompt, args.cover, args.output, args.k)

if __name__ == "__main__":
    main()