"""
ParaSync Torchaudio Forced Alignment Module
使用 torchaudio 的 forced alignment 功能
基于 wav2vec2 模型，效果更好且轻量
"""

import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AlignmentSegment:
    """对齐结果段"""
    token: str
    start_time: float
    end_time: float
    confidence: float


class TorchaudioAligner:
    """
    Torchaudio 强制对齐器
    使用 Wav2Vec2 模型通过 torchaudio 的 forced_align 功能
    """

    # 预训练模型映射
    MODEL_MAP = {
        "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "en": "facebook/wav2vec2-base-960h",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        language: str = "zh",
        device: Optional[str] = None
    ):
        """
        初始化对齐器

        Args:
            model_name: 模型名称（None 则使用 language 对应的默认模型）
            language: 语言代码
            device: 计算设备
        """
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or self.MODEL_MAP.get(language, self.MODEL_MAP["en"])

        self.model = None
        self.processor = None
        self.labels = None

        self._load_model()

    def _load_model(self):
        """加载 Wav2Vec2 模型"""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        print(f"[Aligner] 正在加载模型: {self.model_name}")

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # 获取标签映射
        self.labels = self.processor.tokenizer.convert_ids_to_tokens(
            list(range(len(self.processor.tokenizer)))
        )

        print(f"[Aligner] 模型加载完成")
        print(f"[Aligner] 词汇表大小: {len(self.labels)}")

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        音频预处理

        Args:
            audio_path: 音频文件路径

        Returns:
            (waveform, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重采样到 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(), sample_rate

    def text_to_tokens(self, text: str) -> List[int]:
        """
        将文本转换为 token IDs
        """
        # 使用 processor 处理文本
        inputs = self.processor(text=text, return_tensors="pt")
        return inputs.input_ids.squeeze().tolist()

    def align(
        self,
        audio_path: str,
        transcript: str
    ) -> List[AlignmentSegment]:
        """
        执行强制对齐

        Args:
            audio_path: 音频文件路径
            transcript: 参考文本

        Returns:
            对齐结果列表
        """
        # 预处理音频
        waveform, sample_rate = self.preprocess_audio(audio_path)
        duration = waveform.shape[0] / sample_rate

        # 处理文本
        inputs = self.processor(
            text=transcript,
            return_tensors="pt"
        )
        target_tokens = inputs.input_ids.squeeze().tolist()

        # 处理音频
        audio_inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        input_values = audio_inputs.input_values.to(self.device)

        # 获取 emission (CTC probabilities)
        with torch.no_grad():
            logits = self.model(input_values).logits
            emissions = torch.log_softmax(logits, dim=-1)

        # 使用 torchaudio 的 forced_align
        emission = emissions[0].cpu()
        tokens = torch.tensor([target_tokens], dtype=torch.long)

        # 执行 forced alignment
        alignments, scores = torchaudio.functional.forced_align(
            emission.unsqueeze(0),
            tokens,
            input_lengths=torch.tensor([emission.shape[0]]),
            target_lengths=torch.tensor([len(target_tokens)]),
            blank=0
        )

        # 转换为时间段
        alignment = alignments[0].tolist()
        score = scores[0].tolist()

        segments = self._alignment_to_segments(
            alignment, target_tokens, transcript, duration
        )

        return segments

    def _alignment_to_segments(
        self,
        alignment: List[int],
        tokens: List[int],
        transcript: str,
        duration: float
    ) -> List[AlignmentSegment]:
        """
        将帧级对齐转换为时间段
        """
        segments = []
        frame_duration = duration / len(alignment)

        # 过滤 tokens 中的特殊 token (pad, etc)
        valid_tokens = [t for t in tokens if t > 0]

        # 构建 token 到字符的映射 - 考虑 subword tokenization
        char_idx = 0
        token_to_char = {}
        for i, token_id in enumerate(valid_tokens):
            # 获取 token 对应的文本
            token_text = self.processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            # 移除特殊前缀 (如 ▁ 或 ##)
            clean_text = token_text.replace('▁', '').replace('##', '').strip()

            if clean_text and char_idx < len(transcript):
                # 查找匹配的字符
                token_to_char[token_id] = transcript[char_idx]
                char_idx += len(clean_text)
            else:
                token_to_char[token_id] = ""

        # 去重并合并连续相同标签
        prev_token = None
        start_frame = 0

        for frame_idx, token_id in enumerate(alignment):
            # 跳过 blank (blank_id = 0)
            if token_id == 0:
                continue

            if token_id != prev_token and prev_token is not None:
                # 保存上一个片段
                end_frame = frame_idx
                seg_duration = (end_frame - start_frame) * frame_duration

                if seg_duration > 0.01:  # 过滤过短的片段
                    char = token_to_char.get(prev_token, "")
                    if char:  # 只添加非空字符
                        segments.append(AlignmentSegment(
                            token=char,
                            start_time=start_frame * frame_duration,
                            end_time=end_frame * frame_duration,
                            confidence=0.9
                        ))

                start_frame = frame_idx

            prev_token = token_id

        # 处理最后一个片段
        if prev_token is not None and prev_token != 0:
            end_frame = len(alignment)
            char = token_to_char.get(prev_token, "")
            if char:
                segments.append(AlignmentSegment(
                    token=char,
                    start_time=start_frame * frame_duration,
                    end_time=end_frame * frame_duration,
                    confidence=0.9
                ))

        return segments


if __name__ == "__main__":
    print("Torchaudio Aligner 模块加载成功")
    print("\n使用方法:")
    print("  from aligner.torchaudio_aligner import TorchaudioAligner")
    print("  aligner = TorchaudioAligner(language='zh')")
    print("  segments = aligner.align('audio.wav', '你好世界')")
