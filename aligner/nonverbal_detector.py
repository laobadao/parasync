"""
ParaSync Nonverbal Event Detector Module
非语言事件检测模块：识别呼吸、笑声、哭声等
"""

import torch
import torchaudio
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class EventType(Enum):
    """非语言事件类型"""
    BREATH = "hx"           # 呼吸声
    LAUGH = "laugh"         # 笑声
    CRY = "cry"             # 哭声
    COUGH = "cough"         # 咳嗽
    SIGH = "sigh"           # 叹息
    SILENCE = "sil"         # 静音/停顿
    FILLED_PAUSE = "fp"     # 填充停顿 (um, uh)


@dataclass
class NonverbalEvent:
    """非语言事件数据类"""
    event_type: EventType
    start_time: float
    end_time: float
    confidence: float
    features: Optional[Dict] = None

    def to_label(self) -> str:
        """转换为标签格式 [hx] [laugh] 等"""
        return f"[{self.event_type.value}]"


class NonverbalEventDetector:
    """
    非语言事件检测器
    支持基于能量/频谱的启发式检测和深度学习模型检测
    """

    def __init__(
        self,
        method: str = "heuristic",  # 'heuristic' or 'model'
        sample_rate: int = 16000,
        device: Optional[str] = None
    ):
        """
        初始化检测器

        Args:
            method: 检测方法 ('heuristic' 或 'model')
            sample_rate: 音频采样率
            device: 计算设备
        """
        self.method = method
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 启发式检测参数
        self.energy_threshold = 0.02
        self.silence_threshold = 0.008  # 降低静音阈值，避免过度检测
        self.min_event_duration = 0.08  # 降低最小持续时间，捕捉短呼吸
        self.breath_threshold = 0.02    # 呼吸检测阈值

        # 模型（如使用深度学习方法）
        self.model = None
        if method == "model":
            self._load_model()

    def _load_model(self):
        """加载预训练的音频事件检测模型"""
        try:
            # 这里可以加载 YAMNet, PANNs 等模型
            import tensorflow_hub as hub
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("[ParaSync] YAMNet 模型加载成功")
        except Exception as e:
            print(f"[ParaSync] 模型加载失败，回退到启发式方法: {e}")
            self.method = "heuristic"

    def detect(
        self,
        audio_path: str,
        events_to_detect: Optional[List[EventType]] = None
    ) -> List[NonverbalEvent]:
        """
        检测音频中的非语言事件

        Args:
            audio_path: 音频文件路径
            events_to_detect: 要检测的事件类型列表，None 表示检测全部

        Returns:
            检测到的事件列表
        """
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze().numpy()

        if self.method == "model" and self.model is not None:
            return self._detect_with_model(waveform, events_to_detect)
        else:
            return self._detect_heuristic(waveform, events_to_detect)

    def _detect_heuristic(
        self,
        waveform: np.ndarray,
        events_to_detect: Optional[List[EventType]] = None
    ) -> List[NonverbalEvent]:
        """
        基于启发式规则的检测
        使用能量、频谱质心、过零率等特征
        """
        events = []
        events_to_detect = events_to_detect or list(EventType)

        # 计算基本特征
        features = self._extract_features(waveform)
        energy = features["energy"]
        spectral_centroid = features["spectral_centroid"]
        zcr = features["zero_crossing_rate"]

        # 帧时长
        hop_length = int(0.01 * self.sample_rate)  # 10ms hop
        frame_duration = hop_length / self.sample_rate

        # 检测静音/停顿
        if EventType.SILENCE in events_to_detect:
            silence_events = self._detect_silence(energy, frame_duration)
            events.extend(silence_events)

        # 检测呼吸声
        if EventType.BREATH in events_to_detect:
            breath_events = self._detect_breath(
                waveform, energy, spectral_centroid, zcr, frame_duration
            )
            events.extend(breath_events)

        # 检测笑声
        if EventType.LAUGH in events_to_detect:
            laugh_events = self._detect_laugh(
                energy, spectral_centroid, zcr, frame_duration
            )
            events.extend(laugh_events)

        # 按时间排序
        events.sort(key=lambda x: x.start_time)

        return events

    def _extract_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """提取音频特征"""
        hop_length = int(0.01 * self.sample_rate)  # 10ms

        # 能量 (RMS)
        rms = librosa.feature.rms(
            y=waveform,
            hop_length=hop_length,
            frame_length=hop_length * 4
        )[0]

        # 频谱质心
        spec_cent = librosa.feature.spectral_centroid(
            y=waveform,
            sr=self.sample_rate,
            hop_length=hop_length
        )[0]

        # 过零率
        zcr = librosa.feature.zero_crossing_rate(
            y=waveform,
            hop_length=hop_length,
            frame_length=hop_length * 4
        )[0]

        # 频谱滚降
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=waveform,
            sr=self.sample_rate,
            hop_length=hop_length
        )[0]

        # 梅尔频谱 (用于笑声检测)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            hop_length=hop_length,
            n_mels=40
        )
        mel_energy = np.mean(mel_spec, axis=0)

        return {
            "energy": rms,
            "spectral_centroid": spec_cent,
            "zero_crossing_rate": zcr,
            "spectral_rolloff": spec_rolloff,
            "mel_energy": mel_energy,
        }

    def _detect_silence(
        self,
        energy: np.ndarray,
        frame_duration: float,
        min_duration: float = 0.2
    ) -> List[NonverbalEvent]:
        """检测静音段"""
        events = []
        is_silence = energy < self.silence_threshold

        start_frame = None
        for i, silent in enumerate(is_silence):
            if silent and start_frame is None:
                start_frame = i
            elif not silent and start_frame is not None:
                duration = (i - start_frame) * frame_duration
                if duration >= min_duration:
                    events.append(NonverbalEvent(
                        event_type=EventType.SILENCE,
                        start_time=start_frame * frame_duration,
                        end_time=i * frame_duration,
                        confidence=1.0 - np.mean(energy[start_frame:i]),
                        features={"avg_energy": float(np.mean(energy[start_frame:i]))}
                    ))
                start_frame = None

        # 处理尾部
        if start_frame is not None:
            duration = (len(is_silence) - start_frame) * frame_duration
            if duration >= min_duration:
                events.append(NonverbalEvent(
                    event_type=EventType.SILENCE,
                    start_time=start_frame * frame_duration,
                    end_time=len(is_silence) * frame_duration,
                    confidence=1.0 - np.mean(energy[start_frame:]),
                    features={"avg_energy": float(np.mean(energy[start_frame:]))}
                ))

        return events

    def _detect_breath(
        self,
        waveform: np.ndarray,
        energy: np.ndarray,
        spectral_centroid: np.ndarray,
        zcr: np.ndarray,
        frame_duration: float
    ) -> List[NonverbalEvent]:
        """
        检测呼吸声
        特征：低频能量、周期性、中等能量水平
        """
        events = []

        # 呼吸声特征：
        # 1. 能量在中等水平
        # 2. 频谱质心较低 (低频为主)
        # 3. 过零率较低
        # 4. 有一定的周期性

        # 归一化特征
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        centroid_norm = spectral_centroid / (self.sample_rate / 2)

        # 改进的呼吸声掩码 - 更宽松的检测条件
        # 使用动态阈值而不是固定值
        energy_median = np.median(energy_norm)
        breath_mask = (
            (energy_norm > energy_median * 0.5) &  # 比中位数稍高
            (energy_norm < energy_median * 2.0) &  # 但不至于太高
            (centroid_norm < 0.25) &  # 稍微放宽频谱限制
            (zcr < np.percentile(zcr, 60))  # 较低的过零率
        )

        # 提取连续段
        events = self._extract_event_segments(
            breath_mask, EventType.BREATH, energy, frame_duration
        )

        return events

    def _detect_laugh(
        self,
        energy: np.ndarray,
        spectral_centroid: np.ndarray,
        zcr: np.ndarray,
        frame_duration: float
    ) -> List[NonverbalEvent]:
        """
        检测笑声
        特征：能量波动大、频谱较宽、过零率中等偏高
        """
        events = []

        # 笑声特征：
        # 1. 能量有周期性波动
        # 2. 频谱质心变化大
        # 3. 过零率较高

        # 计算能量变化率 (笑声通常是爆发性的)
        energy_diff = np.abs(np.diff(energy, prepend=energy[0]))
        energy_variation = np.convolve(energy_diff, np.ones(5)/5, mode='same')

        # 归一化
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)

        # 笑声掩码
        laugh_mask = (
            (energy_norm > 0.3) &
            (energy_variation > np.percentile(energy_variation, 70)) &
            (zcr > np.percentile(zcr, 40))
        )

        events = self._extract_event_segments(
            laugh_mask, EventType.LAUGH, energy, frame_duration
        )

        return events

    def _extract_event_segments(
        self,
        mask: np.ndarray,
        event_type: EventType,
        energy: np.ndarray,
        frame_duration: float,
        min_duration: float = 0.15
    ) -> List[NonverbalEvent]:
        """从掩码中提取事件段"""
        events = []
        start_frame = None

        for i, is_active in enumerate(mask):
            if is_active and start_frame is None:
                start_frame = i
            elif not is_active and start_frame is not None:
                duration = (i - start_frame) * frame_duration
                if duration >= min_duration:
                    events.append(NonverbalEvent(
                        event_type=event_type,
                        start_time=start_frame * frame_duration,
                        end_time=i * frame_duration,
                        confidence=float(np.mean(energy[start_frame:i])),
                    ))
                start_frame = None

        # 处理尾部
        if start_frame is not None:
            duration = (len(mask) - start_frame) * frame_duration
            if duration >= min_duration:
                events.append(NonverbalEvent(
                    event_type=event_type,
                    start_time=start_frame * frame_duration,
                    end_time=len(mask) * frame_duration,
                    confidence=float(np.mean(energy[start_frame:])),
                ))

        return events

    def _detect_with_model(
        self,
        waveform: np.ndarray,
        events_to_detect: Optional[List[EventType]] = None
    ) -> List[NonverbalEvent]:
        """
        使用深度学习模型检测
        (需要安装 tensorflow/yamnet)
        """
        # 这里实现 YAMNet 或其他模型的调用
        # 简化版本，实际实现需要处理模型输入输出
        events = []
        return events


def merge_alignments(
    phoneme_segments: List,
    event_segments: List[NonverbalEvent],
    overlap_threshold: float = 0.05,
    preserve_phonemes: bool = True
) -> List[Dict]:
    """
    融合音素对齐与非语言事件

    核心逻辑：
    1. 当非语言事件与音素重叠时，优先保留非语言事件
    2. 静音事件特殊处理：不覆盖长音素，而是分割
    3. 调整音素时间戳以避免冲突
    4. 生成统一的时间轴

    Args:
        phoneme_segments: 音素对齐结果
        event_segments: 非语言事件列表
        overlap_threshold: 重叠判定阈值（秒）
        preserve_phonemes: 是否优先保留音素（避免静音过度覆盖）

    Returns:
        融合后的对齐结果
    """
    merged = []

    # 合并所有段
    all_segments = []

    for seg in phoneme_segments:
        all_segments.append({
            "type": "phoneme",
            "token": seg.token,
            "start": seg.start_time,
            "end": seg.end_time,
            "confidence": seg.confidence,
        })

    for evt in event_segments:
        all_segments.append({
            "type": "event",
            "token": evt.to_label(),
            "start": evt.start_time,
            "end": evt.end_time,
            "confidence": evt.confidence,
            "event_type": evt.event_type.value,
            "is_silence": evt.event_type == EventType.SILENCE,
        })

    # 按开始时间排序
    all_segments.sort(key=lambda x: x["start"])

    # 解决重叠
    resolved = []
    for seg in all_segments:
        if not resolved:
            resolved.append(seg)
            continue

        last = resolved[-1]

        # 检查重叠
        if seg["start"] < last["end"] - overlap_threshold:
            # 有重叠
            overlap_duration = last["end"] - seg["start"]
            last_duration = last["end"] - last["start"]

            # 特殊处理：静音不覆盖长音素
            is_silence_event = seg.get("is_silence", False) or seg.get("event_type") == "sil"

            if seg["type"] == "event" and last["type"] == "phoneme":
                if is_silence_event and preserve_phonemes:
                    # 静音事件：如果音素较长(>0.5s)，分割音素而不是覆盖
                    if last_duration > 0.5 and overlap_duration < last_duration * 0.5:
                        # 分割音素：创建两个音素段，中间插入静音
                        mid_point = seg["start"] + (seg["end"] - seg["start"]) / 2
                        # 缩短前一个音素
                        last["end"] = seg["start"]
                        # 添加静音
                        resolved.append(seg)
                        # 添加分割后的后半段音素（复制）
                        if last["end"] - last["start"] > 0.02:  # 确保有有效时长
                            pass  # 简化为只缩短前一个音素
                    else:
                        # 正常截断
                        last["end"] = seg["start"]
                else:
                    # 非静音事件：优先保留事件，截断前一个音素
                    last["end"] = seg["start"]

                if last["end"] > last["start"]:
                    resolved.append(seg)

            elif seg["type"] == "phoneme" and last["type"] == "event":
                # 前一个事件优先，调整当前音素开始时间
                seg["start"] = last["end"]
                if seg["end"] > seg["start"]:
                    resolved.append(seg)
            else:
                # 同类型：保留置信度高的
                if seg.get("confidence", 0) > last.get("confidence", 0):
                    resolved[-1] = seg
        else:
            resolved.append(seg)

    return resolved


if __name__ == "__main__":
    print("ParaSync NonverbalEventDetector 模块加载成功")
