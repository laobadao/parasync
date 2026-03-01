"""
TTS Generator Module Tests
ChatTTS 语音生成器测试
"""

import pytest
import soundfile as sf
from pathlib import Path

from parasync.tts_generator import ChatTTSGenerator, TTSUtterance, generate_test_audio


@pytest.mark.tts
class TestChatTTSGenerator:
    """ChatTTS 生成器测试类"""

    def test_initialization(self):
        """测试生成器初始化"""
        generator = ChatTTSGenerator(compile=False)
        assert generator.device in ["cuda", "cpu"]
        assert generator.sample_rate == 24000
        assert generator.model is None  # 延迟加载

    def test_supported_tags(self):
        """测试支持的标签列表"""
        assert "[uv_break]" in ChatTTSGenerator.SUPPORTED_TAGS
        assert "[laugh]" in ChatTTSGenerator.SUPPORTED_TAGS
        assert ChatTTSGenerator.SUPPORTED_TAGS["[uv_break]"] == "呼吸/停顿"

    def test_generate_basic(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试基本语音生成"""
        output_path = temp_output_dir / "test_basic.wav"

        result = chattts_generator.generate(
            text="你好世界",
            output_path=str(output_path)
        )

        assert result.exists()
        assert result.stat().st_size > 0

        # 验证音频格式
        info = sf.info(result)
        assert info.samplerate == 24000
        assert info.duration > 0

    def test_generate_with_break(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试带停顿标签的语音生成"""
        output_path = temp_output_dir / "test_break.wav"

        result = chattts_generator.generate(
            text="你好，[uv_break]世界",
            output_path=str(output_path)
        )

        assert result.exists()

        # 验证音频可以被读取
        data, sr = sf.read(result)
        assert sr == 24000
        assert len(data) > 0

    def test_generate_with_laugh(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试带笑声标签的语音生成"""
        output_path = temp_output_dir / "test_laugh.wav"

        result = chattts_generator.generate(
            text="你好[laugh]",
            output_path=str(output_path)
        )

        assert result.exists()
        info = sf.info(result)
        assert info.duration > 0

    def test_generate_with_combined_tags(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试组合标签的语音生成"""
        output_path = temp_output_dir / "test_combined.wav"

        result = chattts_generator.generate(
            text="我是赵君君，[uv_break][laugh]。",
            output_path=str(output_path)
        )

        assert result.exists()

    @pytest.mark.slow
    def test_generate_test_suite(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试批量测试套件生成"""
        results = chattts_generator.generate_test_suite(
            output_dir=str(temp_output_dir),
            speaker_name="test"
        )

        # 验证所有测试用例都已生成
        assert "basic" in results
        assert "with_break" in results
        assert "with_laugh" in results
        assert "combined" in results
        assert "multiple_breaks" in results

        # 验证返回类型
        for name, utterance in results.items():
            assert isinstance(utterance, TTSUtterance)
            assert utterance.audio_path.exists()
            assert utterance.duration > 0
            assert utterance.sample_rate == 24000

            # 验证标签提取
            if "break" in name:
                assert "[uv_break]" in utterance.tags

    def test_generate_with_events(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试在指定位置插入事件标签"""
        output_path = temp_output_dir / "test_events.wav"

        events = [
            ("词后:赵君君", "[laugh]"),
            ("结尾", "[uv_break]"),
        ]

        result = chattts_generator.generate_with_events(
            base_text="我是赵君君",
            events=events,
            output_path=str(output_path)
        )

        assert result.exists()

    def test_get_audio_duration(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试获取音频时长功能"""
        output_path = temp_output_dir / "test_duration.wav"

        chattts_generator.generate(
            text="测试时长",
            output_path=str(output_path)
        )

        duration = chattts_generator.get_audio_duration(str(output_path))
        assert duration > 0

        # 验证与 soundfile 读取的结果一致
        info = sf.info(output_path)
        assert abs(duration - info.duration) < 0.001

    def test_custom_sample_rate(self, chattts_generator: ChatTTSGenerator, temp_output_dir: Path):
        """测试自定义采样率"""
        output_path = temp_output_dir / "test_sr.wav"

        result = chattts_generator.generate(
            text="测试采样率",
            output_path=str(output_path),
            sample_rate=16000  # ChatTTS 实际输出仍为 24000，但会被重采样
        )

        assert result.exists()


class TestGenerateTestAudio:
    """便捷函数测试"""

    @pytest.mark.tts
    def test_generate_test_audio_function(self, temp_output_dir: Path):
        """测试便捷函数"""
        output_path = temp_output_dir / "convenience_test.wav"

        result = generate_test_audio(
            text="测试便捷函数",
            output_path=str(output_path),
            compile=False
        )

        assert result.exists()
        assert result.stat().st_size > 0


class TestTTSUtterance:
    """TTSUtterance 数据类测试"""

    def test_dataclass_creation(self, temp_output_dir: Path):
        """测试数据类创建"""
        utterance = TTSUtterance(
            text="测试文本",
            audio_path=temp_output_dir / "test.wav",
            duration=2.5,
            sample_rate=24000,
            tags=["[uv_break]"]
        )

        assert utterance.text == "测试文本"
        assert utterance.duration == 2.5
        assert utterance.sample_rate == 24000
        assert "[uv_break]" in utterance.tags
