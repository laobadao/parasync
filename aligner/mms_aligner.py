"""
ParaSync MMS FA (Forced Alignment) Module
使用 Facebook MMS 模型进行强制对齐
"""

import os
import re
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


def chinese_to_pinyin(text: str) -> str:
    """将中文转换为拼音（MMS-1B 使用拉丁字母表示）"""
    from pypinyin import pinyin, Style

    # 转换为拼音，不带声调
    py_list = pinyin(text, style=Style.TONE3, strict=False)
    # 展平并连接
    result = ' '.join([item[0] for item in py_list])
    return result


@dataclass
class MMSAlignmentSegment:
    """MMS 对齐结果段"""
    token: str
    start_time: float
    end_time: float
    confidence: float


class MMSAligner:
    """
    MMS 强制对齐器
    基于 facebook/mms-1b-all 模型
    """

    # 模型和字典路径
    DEFAULT_MODEL_PATH = "models/mms/mms1b_all.pt"
    DEFAULT_DICT_URL = "https://dl.fbaipublicfiles.com/mms/asr/dict/mms1b_all_dict.txt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        dict_path: Optional[str] = None,
        device: Optional[str] = None,
        language: str = "zho"
    ):
        """
        初始化 MMS 对齐器

        Args:
            model_path: 模型文件路径
            dict_path: 字典文件路径
            device: 计算设备
            language: 语言代码 (zho=中文, eng=英文)
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.dict_path = dict_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language

        self.model = None
        self.task = None
        self.label_dir = None

        self._check_and_download()
        self._load_model()

    def _check_and_download(self):
        """检查并下载必要文件"""
        import urllib.request
        import ssl

        # 检查模型文件
        if not Path(self.model_path).exists():
            print(f"⚠️  模型文件不存在: {self.model_path}")
            print(f"   请运行: python scripts/download_mms_model.py")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # 尝试下载字典（可选，模型本身已包含字典信息）
        if self.dict_path is None:
            cache_dir = Path.home() / ".cache" / "parasync" / "mms"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dict_file = cache_dir / "mms1b_all_dict.txt"

            if not dict_file.exists():
                try:
                    print(f"📥 尝试下载字典文件...")
                    # 创建 SSL 上下文（处理某些证书问题）
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    with urllib.request.urlopen(self.DEFAULT_DICT_URL, context=ssl_context, timeout=30) as response:
                        with open(dict_file, 'wb') as f:
                            f.write(response.read())
                    print(f"   保存到: {dict_file}")
                except Exception as e:
                    print(f"   字典下载失败（可忽略）: {e}")
                    print(f"   使用模型内置字典")

            if dict_file.exists():
                self.dict_path = str(dict_file)

    def _load_model(self):
        """加载 MMS 模型"""
        from fairseq import checkpoint_utils

        print(f"[MMS] 正在加载模型: {self.model_path}")
        print(f"[MMS] 设备: {self.device}")

        # 加载模型
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [self.model_path],
            arg_overrides={
                "data": str(Path(self.model_path).parent),
            }
        )

        self.model = models[0]
        self.model.to(self.device)
        self.model.eval()
        self.task = task

        # 获取语言列表
        self.lang_list = self._get_lang_list()
        print(f"[MMS] 模型加载完成，支持 {len(self.lang_list)} 种语言")

    def _get_lang_list(self) -> List[str]:
        """获取支持的语言列表"""
        try:
            # 从字典文件解析
            langs = set()
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            lang = parts[0].split('_')[0]
                            langs.add(lang)
            return sorted(list(langs))
        except Exception:
            # 默认返回常见语言
            return ['zho', 'eng', 'jpn', 'kor', 'deu', 'fra', 'spa']

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        音频预处理

        Args:
            audio_path: 音频文件路径

        Returns:
            (waveform, sample_rate): 音频张量和采样率
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重采样到 16kHz (MMS 要求)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, sample_rate

    def text_to_tokens(self, text: str) -> Tuple[List[int], List[str]]:
        """
        将文本转换为模型 tokens

        Args:
            text: 输入文本

        Returns:
            (token IDs 列表, 原始字符列表)
        """
        from pypinyin import pinyin, Style

        # 使用任务的 target dictionary
        dictionary = self.task.target_dictionary

        # 中文转换为拼音（MMS-1B 使用拉丁字母）
        if self.language == "zho":
            # 按字符转换为拼音
            py_list = pinyin(text, style=Style.TONE3, strict=False)
            chars = [item[0] for item in py_list]
        else:
            # 英文：按空格分割
            chars = text.lower().split()

        tokens = []
        valid_chars = []
        for char in chars:
            # 查找字符在字典中的索引
            idx = dictionary.index(char)
            if idx != dictionary.unk():
                tokens.append(idx)
                valid_chars.append(char)
            else:
                # 尝试逐个字符查找
                for c in char:
                    idx = dictionary.index(c)
                    if idx != dictionary.unk():
                        tokens.append(idx)
                        valid_chars.append(c)

        return tokens, valid_chars

    def align(
        self,
        audio_path: str,
        transcript: str,
        output_format: str = "segment"
    ) -> List[MMSAlignmentSegment]:
        """
        执行强制对齐

        Args:
            audio_path: 音频文件路径
            transcript: 参考文本
            output_format: 输出格式

        Returns:
            对齐结果列表
        """
        # 预处理音频 - 获取原始波形
        waveform, sample_rate = self.preprocess_audio(audio_path)
        waveform = waveform.to(self.device)

        # MMS/Wav2Vec2 模型期望原始波形输入 [batch, length]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, length]

        # 准备文本
        tokens, valid_chars = self.text_to_tokens(transcript)
        if not tokens:
            print("警告: 没有有效的 tokens")
            return []
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # 获取长度
        src_lengths = torch.tensor([waveform.shape[1]], dtype=torch.long).to(self.device)
        tgt_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(self.device)

        # 执行对齐
        with torch.no_grad():
            # 获取 emission (CTC probabilities)
            # 注意：MMS-1B 模型使用适配器层，需要特定的 corpus_key 配置
            # 当前模型存在适配器维度不匹配问题，建议使用 TorchaudioAligner
            try:
                net_output = self.model(source=waveform, padding_mask=None, corpus_key=['eng'])
                encoder_out = net_output["encoder_out"]  # T x B x C
                encoder_out = encoder_out.transpose(0, 1)  # B x T x C
                emissions = self.model.get_logits(encoder_out)
            except Exception as e:
                print(f"[MMS] 模型推理失败: {e}")
                print("[MMS] 提示：当前 MMS-1B 模型有适配器维度问题")
                print("[MMS] 建议使用 TorchaudioAligner 替代")
                raise
            emissions = torch.log_softmax(emissions, dim=-1)

        # CTC 强制对齐
        alignment = self._ctc_forced_align(
            emissions[0].cpu(),
            tokens,
            blank_id=0
        )

        # 转换为时间段
        segments = self._alignment_to_segments(
            alignment,
            tokens,
            valid_chars,
            waveform.shape[1] / sample_rate
        )

        return segments

    def _ctc_forced_align(
        self,
        emissions: torch.Tensor,
        tokens: List[int],
        blank_id: int = 0
    ) -> List[int]:
        """
        CTC 强制对齐算法 (Viterbi)

        Args:
            emissions: CTC emission probabilities [T, V]
            tokens: 目标 token IDs
            blank_id: blank token ID

        Returns:
            对齐路径
        """
        T, V = emissions.shape
        L = len(tokens)

        if L == 0:
            return []

        # 构建 CTC 序列 (插入 blank)
        ctc_tokens = []
        for tid in tokens:
            ctc_tokens.extend([blank_id, tid])
        ctc_tokens.append(blank_id)

        L_ctc = len(ctc_tokens)

        # 动态规划
        dp = torch.full((L_ctc, T), float('-inf'))
        backtrace = torch.zeros((L_ctc, T), dtype=torch.long)

        # 初始化
        dp[0, 0] = emissions[0, ctc_tokens[0]]
        if ctc_tokens[1] != ctc_tokens[0]:
            dp[1, 0] = emissions[0, ctc_tokens[1]]

        # DP 递推
        for t in range(1, T):
            for i in range(L_ctc):
                # 保持当前标签
                score = dp[i, t-1] + emissions[t, ctc_tokens[i]]
                best_prev = i

                # 从前一个标签转移
                if i > 0:
                    trans_score = dp[i-1, t-1] + emissions[t, ctc_tokens[i]]
                    if trans_score > score:
                        score = trans_score
                        best_prev = i - 1

                # 跳过 blank
                if i > 1 and ctc_tokens[i] != ctc_tokens[i-2]:
                    skip_score = dp[i-2, t-1] + emissions[t, ctc_tokens[i]]
                    if skip_score > score:
                        score = skip_score
                        best_prev = i - 2

                dp[i, t] = score
                backtrace[i, t] = best_prev

        # 回溯
        alignment = []
        best_end = L_ctc - 1 if dp[L_ctc-1, T-1] > dp[L_ctc-2, T-1] else L_ctc - 2

        t = T - 1
        i = best_end

        while t >= 0:
            alignment.append(ctc_tokens[i])
            i = backtrace[i, t].item()
            t -= 1

        alignment.reverse()
        return alignment

    def _alignment_to_segments(
        self,
        alignment: List[int],
        tokens: List[int],
        transcript: str,
        duration: float
    ) -> List[MMSAlignmentSegment]:
        """
        将帧级对齐转换为时间段
        """
        segments = []
        frame_duration = duration / len(alignment)

        # 去重并合并连续相同标签
        prev_token = None
        start_frame = 0

        token_idx = 0

        for frame_idx, token_id in enumerate(alignment):
            if token_id == 0:  # skip blank
                continue

            if token_id != prev_token and prev_token is not None:
                # 保存上一个片段
                end_frame = frame_idx

                if token_idx <= len(transcript):
                    char = transcript[token_idx - 1] if token_idx > 0 else ""
                    segments.append(MMSAlignmentSegment(
                        token=char,
                        start_time=start_frame * frame_duration,
                        end_time=end_frame * frame_duration,
                        confidence=0.9  # MMS 不直接提供置信度
                    ))

                start_frame = frame_idx
                token_idx += 1

            prev_token = token_id

        # 处理最后一个片段
        if prev_token is not None and prev_token != 0:
            if token_idx <= len(transcript):
                char = transcript[token_idx - 1] if token_idx > 0 else ""
                segments.append(MMSAlignmentSegment(
                    token=char,
                    start_time=start_frame * frame_duration,
                    end_time=len(alignment) * frame_duration,
                    confidence=0.9
                ))

        return segments

    @staticmethod
    def download_model(output_dir: str = "models/mms"):
        """
        下载 MMS 模型
        """
        import urllib.request
        from tqdm import tqdm

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_url = "https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt"
        model_file = output_path / "mms1b_all.pt"

        if model_file.exists():
            print(f"✅ 模型已存在: {model_file}")
            return str(model_file)

        print(f"📥 下载 MMS 模型...")
        print(f"   模型大小: ~3.5GB")
        print(f"   保存位置: {model_file}")

        # 下载并显示进度
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="mms1b_all.pt") as t:
            urllib.request.urlretrieve(
                model_url,
                model_file,
                reporthook=t.update_to
            )

        print(f"✅ 下载完成: {model_file}")
        return str(model_file)


if __name__ == "__main__":
    print("MMS Aligner 模块加载成功")
    print(f"使用方法:")
    print(f"  from aligner.mms_aligner import MMSAligner")
    print(f"  aligner = MMSAligner()")
    print(f"  segments = aligner.align('audio.wav', '你好世界')")
