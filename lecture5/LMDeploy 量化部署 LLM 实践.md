#  LMDeploy 量化部署 LLM 实践

## 环境配置

```bash
# 安装环境
studio-conda -t lmdeploy -o pytorch-2.1.2

# 开启环境
conda activate lmdeploy

# 下载依赖
pip install lmdeploy[all]==0.3.0
```



## 使用LMDeploy部署模型并与之对话

```bash
# lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]
lmdeploy chat /root/internlm2-chat-1_8b
```

![image-20240602171317735](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture5/image-20240602171317735.png)

![image-20240602171325267](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture5/image-20240602171325267.png)



## 量化

量化：降低参数或中间结果的精度，进而节省空间，实现性能的提升



- 计算密集：推理过程中，大部分时间耗费在计算上 => 采用更快的硬件提升速度
- 访存密集：推理过程中，大部分时间耗费在访问内存上 => 减少访存次数，提高计算访存比，降低访存量

常见的 LLM 通常是访存密集的，可以通过量化降低显存的占用，进而提升性能



- `KV8` ：缓存技术，可以通过查询缓存来避免对问题进行重复推理，进而提升速度
- `W4A16`：将 `FP16` 权重转换为 `INT4` ，将显存占用下降为之前的 `1/4`



### 运行量化后的模型

```bash
# 安装依赖库
pip install einops==0.7.0

# W4A16 量化，较慢
lmdeploy lite auto_awq \
   /root/internlm2-chat-1_8b \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/internlm2-chat-1_8b-4bit

# 0.4 缓存 + W416A 模型部署
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4
```

![image-20240602173851537](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture5/image-20240602173851537.png)

![image-20240602173914511](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture5/image-20240602173914511.png)

使用 `LMDeploy` 部署，并使用量化后，速度和显存占比均有优化



## 小结

### 推理引擎更快的原因

结合上面询问 `internlm2-chat-1_8b` 的内容，可以发现，推理引擎的主要作用是优化模型对资源的利用，提高资源的利用率，进而提升模型的推理速度