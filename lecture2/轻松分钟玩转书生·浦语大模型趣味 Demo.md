# 轻松分钟玩转书生·浦语大模型趣味 Demo

## 部署 `InternLM2-Chat-1.8B` 模型

### 环境配置

#### 创建开发机

![image-20240526193544688](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526193544688.png)

![image-20240526193621317](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526193621317.png)

#### 配置环境，耗时长

```bash
studio-conda -o internlm-base -t demo
```

```bash
# 与 studio-conda 等效的配置方案
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### 运行环境

```bash
conda activate demo
```

#### 按照环境需要的包

```bash
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```



### 下载模型

#### 创建目录

```bash
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```

#### 修改python文件

`download_mini.py`

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```



`cli_demo.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```

#### 下载模型参数

```bash
python /root/demo/download_mini.py
```

#### 运行模型

```bash
conda activate demo
python /root/demo/cli_demo.py
```



### 使用模型

![image-20240526195053566](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526195053566.png)



## 部署`八戒-Chat-1.8B`模型

### 前置条件：环境已配置且已运行环境

>配置环境，耗时较长
>
>```bash
>studio-conda -o internlm-base -t demo
>
># 与 studio-conda 等效的配置方案
># conda create -n demo python==3.10 -y
># conda activate demo
># conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
>```
>
>
>
>运行环境
>
>```bash
>conda activate demo
>```



### 创建目录

```bash
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
```

### 下载模型

```bash
python /root/Tutorial/helloworld/bajie_download.py
```

### 运行模型

```bash
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```

如果执行命令时出现如下错误，检查是否运行了环境

![image-20240526195841570](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526195841570.png)



### 配置端口映射

#### 打开`powershell`

`win+R` 输入 `powershell`

![image-20240526200009485](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526200009485.png	)



#### 查询开发机端口

![image-20240526200133642](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526200133642.png)



#### 开启端口映射

下述命令在本地机器上执行

```bash
# 从本地使用 ssh 连接 studio 端口
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 33330
```

输入上图中的密码



#### 访问地址

![image-20240526200337262](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526200337262.png)

输入密码后，访问命令中输入的端口

http://127.0.0.1:6006/



### 一些问题

#### repo_name

![image-20240526201057261](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526201057261.png)

原因：忘记执行下载模型的命令

```bash
python /root/Tutorial/helloworld/bajie_download.py
```

运行后在启动模型



![image-20240526202933745](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture2/image-20240526202933745.png)