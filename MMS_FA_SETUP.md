# MMS FA (Forced Alignment) 设置指南

## 当前状态
使用的是 wav2vec2 通用模型，对齐精度有限。

## MMS FA 优势
- 专门优化的 forced alignment 模型
- 更好的音素边界检测
- 支持 1000+ 语言

## 需要完成的设置

### 1. 安装 fairseq
```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install -e .
```

### 2. 下载 MMS FA 模型
```bash
# 下载模型和字典
wget https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt
wget https://dl.fbaipublicfiles.com/mms/asr/dict/mms1b_all_dict.txt
```

### 3. 准备数据格式
需要生成以下文件：
- `audio.wav` - 音频文件
- `transcript.txt` - 文本内容
- `manifest.tsv` - 清单文件

### 4. 运行对齐
```bash
python align.py \
  --audio audio.wav \
  --text transcript.txt \
  --model mms1b_all.pt \
  --dict mms1b_all_dict.txt \
  --output result.TextGrid
```

## 下一步行动

请确认：
1. 是否可以安装 fairseq？（需要编译，可能需要 C++ 环境）
2. 是否接受模型下载（约 3.5GB）？
3. 是否需要我实现自动下载和设置脚本？
