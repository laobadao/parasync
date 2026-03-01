"""
ParaSync ASR Recognizer Module
语音识别模块，使用 Whisper 或 MMS 模型
"""

import torch
import torchaudio
from typing import Optional
from dataclasses import dataclass


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    confidence: float
    language: str


class ASRRecognizer:
    """
    语音识别器
    支持 Whisper 和 MMS 模型
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        language: str = "zh"
    ):
        """
        初始化 ASR

        Args:
            model_name: 模型名称
                - "openai/whisper-base" (推荐中文)
                - "openai/whisper-small"
                - "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            device: 计算设备
            language: 语言代码
        """
        self.model_name = model_name
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """加载 ASR 模型"""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        print(f"[ASR] 正在加载模型: {self.model_name}")

        if "whisper" in self.model_name.lower():
            # Whisper 模型
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
            )
            self.is_whisper = True
        else:
            # Wav2Vec2 模型
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
            )
            self.is_whisper = False

        print(f"[ASR] 模型加载完成")

    def recognize(
        self,
        audio_path: str,
        return_timestamps: bool = False
    ) -> ASRResult:
        """
        识别音频

        Args:
            audio_path: 音频文件路径
            return_timestamps: 是否返回时间戳

        Returns:
            ASRResult 识别结果
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重采样到 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 准备输入
        audio_data = waveform.squeeze().numpy()

        # 识别
        if self.is_whisper:
            result = self.pipe(
                audio_data,
                return_timestamps=return_timestamps,
                generate_kwargs={"language": self.language}
            )
        else:
            result = self.pipe(audio_data, return_timestamps=return_timestamps)

        text = result.get("text", "") if isinstance(result, dict) else str(result)

        return ASRResult(
            text=text.strip(),
            confidence=result.get("confidence", 0.9) if isinstance(result, dict) else 0.9,
            language=self.language
        )


def recognize_audio(audio_path: str, language: str = "zh") -> str:
    """
    便捷函数：识别音频文件

    Args:
        audio_path: 音频路径
        language: 语言代码

    Returns:
        识别的文本
    """
    recognizer = ASRRecognizer(language=language)
    result = recognizer.recognize(audio_path)
    return result.text


if __name__ == "__main__":
    print("ASR Recognizer 模块加载成功")
