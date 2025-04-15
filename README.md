# LLMSteg
基于大语言模型的文本隐写算法

## 1. 配置
1. 安装依赖

    ```shell
    conda create --name llmsteg python=3.11
    conda activate llmsteg
    pip install -r requirements.txt
    ```
2. 配置环境变量

    在`.env`文件中配置环境变量`MODEL_DIR`。该变量应为模型路径，或者能够被[modelscope](https://github.com/modelscope/modelscope)识别的模型名称。默认为`Qwen/Qwen2.5-3B-Instruct`

## 2. 使用
```shell
$ python main.py --help                                                                                                       
usage: main.py [-h] --encode_decode {0,1} [--k K] --prompt PROMPT [--secret SECRET] [--cover COVER] --output OUTPUT

options:
  -h, --help            show this help message and exit
  --encode_decode {0,1}
                        0 for encode, 1 for decode
  --k K                 Value of k
  --prompt PROMPT       File path to read prompt
  --secret SECRET       File path to read secret data, used when --encode_decode=0
  --cover COVER         File path to read response, used when --encode_decode=1
  --output OUTPUT       File path to write output
```
