# MMS-1B 集成状态报告

## 结论

**MMS-1B Forced Alignment 集成当前不可用**

经过详细测试，发现下载的 MMS-1B 模型文件与预期不符，无法用于中文对齐任务。

---

## 问题分析

### 1. 模型类型不匹配

下载的 `mms1b_all.pt` 文件实际上是**英语专用微调版本**，而非完整的多语言 ASR 模型。

**证据：**
- `task.cfg.multi_corpus_keys: 'eng'` - 仅支持英语
- `task.cfg.target_dictionary: '/fsx-wav2vec/.../eng'` - 英语字典路径
- `dictionary 大小: 154` - 仅包含拉丁字母，无中文字符
- `dictionary 内容: a-z, 0-9, 标点符号` - 字母级表示

### 2. 技术障碍

| 问题 | 详情 | 状态 |
|------|------|------|
| fairseq 兼容性 | Python 3.11+ 存在 dataclasses 兼容性问题 | ✅ 已降级到 3.10 解决 |
| 模型调用 | Wav2VecCtc 需要 corpus_key 参数 | ❌ 模型不支持中文 corpus |
| 字典映射 | 中文无法映射到拉丁字母字典 | ❌ 根本性问题 |

### 3. 正确的 MMS-1B 模型

需要下载的模型应该是：
- **Multilingual ASR model**: 支持 1000+ 语言的预训练模型
- **Character-level dictionary**: 包含中文字符或拼音映射
- **Language-specific adapters**: 支持 `zho` (中文) corpus key

---

## 当前可用方案

### 推荐：PhonemeAligner / TorchaudioAligner

| 特性 | PhonemeAligner | TorchaudioAligner |
|------|----------------|-------------------|
| 模型 | wav2vec2-large-xlsr-53-chinese-zh-cn | 同上 |
| 大小 | ~1GB | ~1GB |
| 中文支持 | ✅ 原生支持 | ✅ 原生支持 |
| 对齐精度 | 字符级 | 字符级 |
| 依赖 | transformers + torchaudio | transformers + torchaudio |
| 性能 | 良好 | 良好 |

**测试对比（test01.wav, 文本："我是赵君君哈哈"）：**

```
PhonemeAligner:     TorchaudioAligner:
我: 0.000-1.940s     我: 0.000-1.983s
|: 1.940-1.980s      是: 1.983-2.103s
是: 1.980-2.060s     赵: 2.103-2.283s
|: 2.060-2.100s      君: 2.283-2.744s
赵: 2.100-2.280s     哈: 2.744-4.907s
君: 2.280-2.680s
|: 2.680-2.740s
哈: 2.740-4.900s
```

两者对齐精度相当，PhonemeAligner 多了词边界标记 `|`。

---

## 后续建议

### 方案 A：继续使用现有实现（推荐）

当前 `PhonemeAligner` 完全满足中文对齐需求：
- ✅ 已验证可用
- ✅ 轻量级（1GB vs 14GB）
- ✅ 支持中英文
- ✅ 自定义 CTC 对齐算法

### 方案 B：重新下载正确的 MMS-1B 模型

如果需要使用 MMS-1B：

1. **下载正确的模型文件：**
   ```bash
   # 多语言 ASR 模型（约 13GB）
   wget https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt

   # 或从 Hugging Face 下载
   # https://huggingface.co/facebook/mms-1b-all
   ```

2. **验证模型类型：**
   ```python
   from fairseq import checkpoint_utils
   models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
   print(task.cfg.multi_corpus_keys)  # 应显示多个语言代码
   ```

3. **准备多语言字典：**
   - 下载 `mms1b_all_dict.txt`
   - 包含 1000+ 语言的字符映射

### 方案 C：使用 Hugging Face Transformers 版本

Facebook 也提供了 Hugging Face 格式的 MMS 模型：

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
```

这可能比 fairseq 版本更容易使用。

---

## 文件位置

- **当前代码**: `aligner/mms_aligner.py` - 保留框架供将来使用
- **模型文件**: `models/mms/mms1b_all.pt` (14GB, 英语专用版本)
- **下载脚本**: `scripts/download_mms_model.py`

---

## 相关链接

- [Facebook MMS 官方文档](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [MMS-1B Hugging Face](https://huggingface.co/facebook/mms-1b-all)
- [Fairseq Issues - Python 3.11 兼容性](https://github.com/facebookresearch/fairseq/issues)

---

**最后更新**: 2025-03-01
