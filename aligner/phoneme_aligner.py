"""
ParaSync Phoneme Aligner Module
基于 MMS-FA 的强制对齐器，支持中英文音素对齐
"""

import torch
import torchaudio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class AlignmentSegment:
    """对齐片段数据类"""
    token: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


class PhonemeAligner:
    """
    基于 MMS (Massively Multilingual Speech) 的强制对齐器
    支持中英文及多种语言的音素级别对齐
    """

    # 语言模型映射表
    LANG_MODEL_MAP = {
        "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "en": "facebook/wav2vec2-base-960h",
        "ja": "facebook/wav2vec2-xlsr-53",
        "ko": "facebook/wav2vec2-xlsr-53",
        "de": "facebook/wav2vec2-xlsr-53",
        "fr": "facebook/wav2vec2-xlsr-53",
        "es": "facebook/wav2vec2-xlsr-53",
        # 更多语言可扩展...
    }

    # 音素字典缓存
    _phoneme_dict_cache: Dict[str, Dict] = {}

    def __init__(
        self,
        lang: str = "zh",
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        初始化对齐器

        Args:
            lang: 语言代码 ('zh', 'en', etc.)
            device: 计算设备 ('cuda', 'cpu', or None for auto)
            sample_rate: 目标采样率，MMS 推荐 16kHz
        """
        self.lang = lang
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型和处理器
        self._load_model()

    def _load_model(self):
        """加载 MMS 模型和相关组件"""
        try:
            from transformers import Wav2Vec2ForCTC, AutoProcessor
        except ImportError:
            raise ImportError(
                "请安装 transformers: pip install transformers"
            )

        model_name = self.LANG_MODEL_MAP.get(self.lang, "facebook/mms-tts-eng")

        print(f"[ParaSync] 正在加载模型: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 获取音素标签映射
        self.id2token = self.processor.tokenizer.get_vocab()
        self.token2id = {v: k for k, v in self.id2token.items()}

        print(f"[ParaSync] 模型加载完成，词汇表大小: {len(self.id2token)}")

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        音频预处理：加载、重采样、归一化

        Args:
            audio_path: 音频文件路径

        Returns:
            预处理后的音频张量
        """
        # 加载音频
        waveform, orig_sr = torchaudio.load(audio_path)

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 重采样到目标采样率
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # 归一化
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        return waveform.squeeze()

    def text_to_tokens(self, text: str) -> List[int]:
        """
        将文本转换为模型 token IDs

        Args:
            text: 输入文本

        Returns:
            Token ID 列表
        """
        # 基础文本清理
        text = self._clean_text(text)

        # 使用处理器进行 tokenization
        inputs = self.processor(text=text, return_tensors="pt")
        token_ids = inputs["input_ids"].squeeze().tolist()

        # 确保是列表格式
        if not isinstance(token_ids, list):
            token_ids = [token_ids]

        return token_ids

    def _clean_text(self, text: str) -> str:
        """文本预处理清理"""
        # 移除多余空白
        text = " ".join(text.split())

        # 语言特定的预处理
        if self.lang == "zh":
            # 中文：在字符间添加空格以帮助对齐
            # 但保留常见多字词
            text = self._segment_chinese(text)
        elif self.lang == "en":
            # 英文：转换为小写
            text = text.lower()

        return text

    def _segment_chinese(self, text: str) -> str:
        """
        中文分词处理
        简单的基于词典的分词，或使用字符级分割
        """
        # 这里可以使用 jieba 进行更精确的分词
        # 为了简化，先使用字符级分割
        try:
            import jieba
            words = jieba.lcut(text)
            return " ".join(words)
        except ImportError:
            # 如果没有 jieba，使用字符级分割
            return " ".join(list(text.replace(" ", "")))

    def compute_emissions(
        self,
        waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        计算 CTC 发射概率

        Args:
            waveform: 预处理后的音频波形

        Returns:
            (emissions, stride): 发射概率和帧步长
        """
        # 将音频移动到设备
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].to(self.device)

        # 前向传播获取 logits
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits

        # 计算 log softmax 获取发射概率
        emissions = torch.log_softmax(logits, dim=-1)

        # 计算帧步长（下采样率）
        stride = waveform.shape[0] // emissions.shape[1]

        return emissions.squeeze(0).cpu(), stride

    def ctc_forced_align(
        self,
        emissions: torch.Tensor,
        token_ids: List[int],
        blank_id: int = 0
    ) -> Tuple[List[int], List[float]]:
        """
        CTC 强制对齐算法

        Args:
            emissions: 发射概率矩阵 [T, V]
            token_ids: 目标 token IDs
            blank_id: 空白标签 ID

        Returns:
            (alignment, scores): 对齐路径和置信度分数
        """
        T, V = emissions.shape
        L = len(token_ids)

        if L == 0:
            return [], []

        # 构建 CTC 标签序列 (插入 blank)
        ctc_tokens = []
        for tid in token_ids:
            ctc_tokens.extend([blank_id, tid])
        ctc_tokens.append(blank_id)

        L_ctc = len(ctc_tokens)

        # 动态规划矩阵
        # dp[i][j] = 到达位置 j 使用标签 i 的最大概率
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

                # 跳过 blank 的特殊处理
                if i > 1 and ctc_tokens[i] != ctc_tokens[i-2]:
                    skip_score = dp[i-2, t-1] + emissions[t, ctc_tokens[i]]
                    if skip_score > score:
                        score = skip_score
                        best_prev = i - 2

                dp[i, t] = score
                backtrace[i, t] = best_prev

        # 回溯获取最优路径
        alignment = []
        scores = []

        # 找到终止位置
        best_end = L_ctc - 1 if dp[L_ctc-1, T-1] > dp[L_ctc-2, T-1] else L_ctc - 2

        t = T - 1
        i = best_end

        while t >= 0:
            alignment.append(ctc_tokens[i])
            scores.append(emissions[t, ctc_tokens[i]].item())
            i = backtrace[i, t].item()
            t -= 1

        alignment.reverse()
        scores.reverse()

        return alignment, scores

    def align_text(
        self,
        audio_path: str,
        transcript: str,
        output_format: str = "segment"
    ) -> List[AlignmentSegment]:
        """
        主要接口：对齐音频与文本

        Args:
            audio_path: 音频文件路径
            transcript: 参考文本
            output_format: 输出格式 ('segment' or 'frame')

        Returns:
            对齐结果列表，包含 (token, start_time, end_time, confidence)
        """
        # 预处理音频
        waveform = self.preprocess_audio(audio_path)

        # 文本转 tokens
        token_ids = self.text_to_tokens(transcript)

        # 计算发射概率
        emissions, stride = self.compute_emissions(waveform)

        # CTC 强制对齐
        alignment, scores = self.ctc_forced_align(emissions, token_ids)

        # 转换为时间戳
        segments = self._alignment_to_segments(
            alignment, scores, token_ids, stride
        )

        return segments

    def _alignment_to_segments(
        self,
        alignment: List[int],
        scores: List[float],
        original_tokens: List[int],
        stride: int
    ) -> List[AlignmentSegment]:
        """
        将帧级对齐转换为时间段

        Args:
            alignment: CTC 对齐路径
            scores: 每帧的置信度
            original_tokens: 原始 token IDs
            stride: 帧步长

        Returns:
            对齐片段列表
        """
        segments = []
        frame_duration = stride / self.sample_rate

        # 去重并合并连续相同标签
        prev_token = None
        start_frame = 0

        for frame_idx, (token_id, score) in enumerate(zip(alignment, scores)):
            # 跳过 blank (假设 blank_id = 0)
            if token_id == 0:
                continue

            if token_id != prev_token and prev_token is not None:
                # 保存上一个片段
                end_frame = frame_idx
                duration = (end_frame - start_frame) * frame_duration

                if duration > 0.01:  # 过滤过短的片段
                    token_str = self.token2id.get(prev_token, "<unk>")
                    segments.append(AlignmentSegment(
                        token=token_str,
                        start_time=start_frame * frame_duration,
                        end_time=end_frame * frame_duration,
                        confidence=np.exp(np.mean(scores[start_frame:end_frame]))
                    ))

                start_frame = frame_idx

            prev_token = token_id

        # 处理最后一个片段
        if prev_token is not None and prev_token != 0:
            end_frame = len(alignment)
            segments.append(AlignmentSegment(
                token=self.token2id.get(prev_token, "<unk>"),
                start_time=start_frame * frame_duration,
                end_time=end_frame * frame_duration,
                confidence=np.exp(np.mean(scores[start_frame:end_frame]))
            ))

        return segments

    def get_word_level_alignment(
        self,
        audio_path: str,
        transcript: str,
        word_delimiter: str = " "
    ) -> List[AlignmentSegment]:
        """
        获取词级别对齐（将音素聚合成词）

        Args:
            audio_path: 音频文件路径
            transcript: 参考文本
            word_delimiter: 词分隔符

        Returns:
            词级别对齐结果
        """
        # 首先获取音素级别对齐
        phone_segments = self.align_text(audio_path, transcript)

        # 按词分组
        words = transcript.split(word_delimiter)
        word_segments = []

        # 这里需要更复杂的映射逻辑
        # 简化版本：按时间均匀分配
        total_duration = phone_segments[-1].end_time if phone_segments else 0
        word_duration = total_duration / len(words) if words else 0

        for i, word in enumerate(words):
            word_segments.append(AlignmentSegment(
                token=word,
                start_time=i * word_duration,
                end_time=(i + 1) * word_duration,
                confidence=0.9  # 占位符
            ))

        return word_segments


# 便捷函数
def align_audio_text(
    audio_path: str,
    transcript: str,
    lang: str = "zh"
) -> List[AlignmentSegment]:
    """
    一键对齐函数

    Example:
        >>> results = align_audio_text("audio.wav", "你好世界", lang="zh")
        >>> for seg in results:
        ...     print(f"{seg.token}: {seg.start_time:.3f}s - {seg.end_time:.3f}s")
    """
    aligner = PhonemeAligner(lang=lang)
    return aligner.align_text(audio_path, transcript)


if __name__ == "__main__":
    # 简单测试
    print("ParaSync PhonemeAligner 模块加载成功")
    print("支持的语音:", list(PhonemeAligner.LANG_MODEL_MAP.keys()))
