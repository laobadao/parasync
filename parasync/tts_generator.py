"""
ParaSync TTS Generator Module
使用 ChatTTS 生成测试语音，支持 [uv_break] [laugh] 等特殊标签
"""

import torch
import ChatTTS
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TTSUtterance:
    """TTS 生成的话语数据类"""
    text: str
    audio_path: Path
    duration: float
    sample_rate: int
    tags: List[str]  # 包含的标签如 [uv_break], [laugh]


class ChatTTSGenerator:
    """
    ChatTTS 语音生成器
    用于生成带特殊标签的测试音频
    """

    # 支持的 ChatTTS 特殊标签
    SUPPORTED_TAGS = {
        "[uv_break]": "呼吸/停顿",
        "[laugh]": "笑声",
        "[lbreak]": "长停顿",
        "[break]": "短停顿",
    }

    def __init__(self, compile: bool = False, device: Optional[str] = None):
        """
        初始化 ChatTTS 生成器

        Args:
            compile: 是否编译模型以获得更快推理速度
            device: 计算设备 ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 24000  # ChatTTS 默认采样率
        self.compile = compile
        self.model = None

    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            print("[ChatTTSGenerator] 正在加载 ChatTTS 模型...")
            self.model = ChatTTS.Chat()
            self.model.load(compile=self.compile)
            print("[ChatTTSGenerator] 模型加载完成")

    def generate(
        self,
        text: str,
        output_path: str,
        sample_rate: Optional[int] = None,
        params_refine_text: Optional[Dict] = None,
        params_infer_code: Optional[Dict] = None,
    ) -> Path:
        """
        生成语音并保存

        Args:
            text: 输入文本，可包含 [uv_break] [laugh] 等标签
            output_path: 输出音频文件路径
            sample_rate: 输出采样率，None 则使用默认 24000
            params_refine_text: 文本精炼参数
            params_infer_code: 代码推理参数

        Returns:
            输出文件路径
        """
        self._load_model()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 默认参数
        params_refine_text = params_refine_text or {
            "prompt": "[oral_2][laugh_0][break_6]"
        }
        params_infer_code = params_infer_code or {
            "prompt": "[speed_5]",
            "temperature": 0.3,
            "max_new_token": 2048,
        }

        # 生成语音
        wavs = self.model.infer(
            [text],
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )

        # 保存音频
        sr = sample_rate or self.sample_rate
        sf.write(output_path, wavs[0].squeeze(), sr)

        print(f"[ChatTTSGenerator] 语音已保存: {output_path}")
        return output_path

    def generate_test_suite(
        self,
        output_dir: str,
        speaker_name: str = "test_speaker"
    ) -> Dict[str, TTSUtterance]:
        """
        生成完整测试套件

        Args:
            output_dir: 输出目录
            speaker_name: 说话人名称，用于文件命名

        Returns:
            测试用例字典，key 为测试名，value 为 TTSUtterance
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 定义测试用例
        test_cases: Dict[str, Tuple[str, List[str]]] = {
            "basic": ("我是赵君君", []),
            "with_break": ("我是赵君君，[uv_break]。", ["[uv_break]"]),
            "with_laugh": ("我是赵君君，[laugh]。", ["[laugh]"]),
            "combined": ("我是赵君君，[uv_break][laugh]。", ["[uv_break]", "[laugh]"]),
            "multiple_breaks": ("我是[uv_break]赵君君，[uv_break][laugh]。", ["[uv_break]", "[laugh]"]),
            "short_sentence": ("你好", []),
            "long_sentence": ("我是赵君君，很高兴认识你，今天天气真好。", []),
        }

        results = {}

        for test_name, (text, tags) in test_cases.items():
            output_path = output_dir / f"{speaker_name}_{test_name}.wav"

            # 生成音频
            self.generate(
                text=text,
                output_path=str(output_path),
            )

            # 获取音频时长
            info = sf.info(output_path)
            duration = info.duration

            results[test_name] = TTSUtterance(
                text=text,
                audio_path=output_path,
                duration=duration,
                sample_rate=self.sample_rate,
                tags=tags,
            )

        print(f"[ChatTTSGenerator] 测试套件生成完成: {len(results)} 条音频")
        return results

    def generate_with_events(
        self,
        base_text: str,
        events: List[Tuple[str, str]],
        output_path: str,
    ) -> Path:
        """
        在指定位置插入事件标签生成音频

        Args:
            base_text: 基础文本
            events: 事件列表，格式为 [(位置描述, 事件标签), ...]
                   位置描述如 "开头", "结尾", "词后:赵君君"
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        text = base_text

        for position, event_tag in events:
            if position == "开头":
                text = event_tag + text
            elif position == "结尾":
                text = text + event_tag
            elif position.startswith("词后:"):
                word = position.split(":")[1]
                text = text.replace(word, word + event_tag, 1)
            elif position.startswith("词前:"):
                word = position.split(":")[1]
                text = text.replace(word, event_tag + word, 1)

        return self.generate(text, output_path)

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        info = sf.info(audio_path)
        return info.duration


def generate_test_audio(
    text: str,
    output_path: str,
    compile: bool = False,
) -> Path:
    """
    便捷函数：生成单条测试音频

    Example:
        >>> generate_test_audio("你好[uv_break]世界", "test.wav")
    """
    generator = ChatTTSGenerator(compile=compile)
    return generator.generate(text, output_path)


if __name__ == "__main__":
    # 简单测试
    print("ChatTTSGenerator 模块测试")

    gen = ChatTTSGenerator(compile=False)

    # 生成单条测试
    gen.generate(
        text="我是赵君君，[uv_break][laugh]。",
        output_path="test_output.wav"
    )

    print("测试完成")
