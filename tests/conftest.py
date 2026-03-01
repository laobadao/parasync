"""
Pytest Configuration and Shared Fixtures
共享测试 fixtures 和配置
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict
import soundfile as sf
import numpy as np

# 导入被测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from parasync.tts_generator import ChatTTSGenerator
from aligner.phoneme_aligner import PhonemeAligner
from aligner.nonverbal_detector import NonverbalEventDetector, EventType
from aligner.textgrid_exporter import TextGridExporter


# ============================================================================
# 路径和目录 Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="function")
def temp_output_dir() -> Generator[Path, None, None]:
    """为每个测试函数创建临时输出目录"""
    temp_dir = tempfile.mkdtemp(prefix="parasync_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """测试数据目录"""
    data_dir = project_root / "tests" / "fixtures"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def audio_samples_dir(test_data_dir: Path) -> Path:
    """音频样本目录"""
    samples_dir = test_data_dir / "audio_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


# ============================================================================
# 测试文本 Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_text_zh() -> str:
    """中文测试文本"""
    return "我是赵君君"


@pytest.fixture(scope="session")
def sample_text_with_break() -> str:
    """带停顿标签的测试文本"""
    return "我是赵君君，[uv_break]。"


@pytest.fixture(scope="session")
def sample_text_with_laugh() -> str:
    """带笑声标签的测试文本"""
    return "我是赵君君，[laugh]。"


@pytest.fixture(scope="session")
def sample_text_combined() -> str:
    """组合标签测试文本"""
    return "我是赵君君，[uv_break][laugh]。"


# ============================================================================
# TTS 生成器 Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def chattts_generator() -> Generator[ChatTTSGenerator, None, None]:
    """
    共享的 ChatTTS 生成器实例（session 级别）

    注意：ChatTTS 模型加载较慢，使用 session 级别 fixture 避免重复加载
    """
    generator = ChatTTSGenerator(compile=False)
    yield generator


@pytest.fixture(scope="module")
def generated_audio_suite(
    chattts_generator: ChatTTSGenerator,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Dict[str, Path], None, None]:
    """
    生成完整测试音频套件（module 级别，只生成一次）

    Returns:
        测试音频路径字典
    """
    output_dir = tmp_path_factory.mktemp("audio_samples")

    test_cases = {
        "basic": "我是赵君君",
        "with_break": "我是赵君君，[uv_break]。",
        "with_laugh": "我是赵君君，[laugh]。",
        "combined": "我是赵君君，[uv_break][laugh]。",
        "multiple_breaks": "我是[uv_break]赵君君，[uv_break][laugh]。",
    }

    audio_paths = {}

    for name, text in test_cases.items():
        output_path = output_dir / f"{name}.wav"
        chattts_generator.generate(text=text, output_path=str(output_path))
        audio_paths[name] = output_path

    yield audio_paths


@pytest.fixture(scope="function")
def sample_audio_file(temp_output_dir: Path) -> Path:
    """生成一个简单的测试音频文件（静音 + 正弦波）"""
    output_path = temp_output_dir / "sample.wav"
    sample_rate = 16000
    duration = 2.0

    # 生成简单的正弦波音频
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 添加一些变化模拟语音
    freq = 440  # A4 音
    audio = np.sin(2 * np.pi * freq * t) * 0.3

    # 添加一些静音段
    silence_samples = int(0.3 * sample_rate)
    audio = np.concatenate([
        np.zeros(silence_samples),
        audio,
        np.zeros(silence_samples)
    ])

    sf.write(output_path, audio, sample_rate)
    return output_path


@pytest.fixture(scope="function")
def silence_audio_file(temp_output_dir: Path) -> Path:
    """生成静音测试音频"""
    output_path = temp_output_dir / "silence.wav"
    sample_rate = 16000
    duration = 2.0

    audio = np.zeros(int(sample_rate * duration))
    sf.write(output_path, audio, sample_rate)
    return output_path


# ============================================================================
# Aligner Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def sample_aligner() -> PhonemeAligner:
    """预初始化的音素对齐器（每个测试函数新建，避免状态污染）"""
    return PhonemeAligner(lang="zh")


@pytest.fixture(scope="function")
def sample_detector() -> NonverbalEventDetector:
    """预初始化的事件检测器"""
    return NonverbalEventDetector(method="heuristic", sample_rate=16000)


@pytest.fixture(scope="function")
def sample_exporter() -> TextGridExporter:
    """预初始化的 TextGrid 导出器"""
    return TextGridExporter()


# ============================================================================
# 测试数据 Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_phoneme_segments():
    """模拟的音素对齐结果"""
    from aligner.phoneme_aligner import AlignmentSegment

    return [
        AlignmentSegment(token="w", start_time=0.0, end_time=0.1, confidence=0.9),
        AlignmentSegment(token="o", start_time=0.1, end_time=0.2, confidence=0.95),
        AlignmentSegment(token="sh", start_time=0.2, end_time=0.3, confidence=0.92),
        AlignmentSegment(token="i", start_time=0.3, end_time=0.4, confidence=0.88),
        AlignmentSegment(token="zh", start_time=0.4, end_time=0.5, confidence=0.9),
        AlignmentSegment(token="ao", start_time=0.5, end_time=0.6, confidence=0.87),
        AlignmentSegment(token="j", start_time=0.6, end_time=0.7, confidence=0.91),
        AlignmentSegment(token="un", start_time=0.7, end_time=0.8, confidence=0.89),
        AlignmentSegment(token="j", start_time=0.8, end_time=0.9, confidence=0.9),
        AlignmentSegment(token="un", start_time=0.9, end_time=1.0, confidence=0.88),
    ]


@pytest.fixture(scope="function")
def mock_event_segments():
    """模拟的非语言事件"""
    from aligner.nonverbal_detector import NonverbalEvent

    return [
        NonverbalEvent(
            event_type=EventType.BREATH,
            start_time=0.35,
            end_time=0.45,
            confidence=0.8,
        ),
        NonverbalEvent(
            event_type=EventType.LAUGH,
            start_time=0.85,
            end_time=0.95,
            confidence=0.75,
        ),
    ]


@pytest.fixture(scope="function")
def mock_merged_segments():
    """模拟的融合后结果"""
    return [
        {"type": "phoneme", "token": "w", "start": 0.0, "end": 0.1, "confidence": 0.9},
        {"type": "phoneme", "token": "o", "start": 0.1, "end": 0.2, "confidence": 0.95},
        {"type": "phoneme", "token": "sh", "start": 0.2, "end": 0.3, "confidence": 0.92},
        {"type": "event", "token": "[hx]", "start": 0.35, "end": 0.45, "confidence": 0.8, "event_type": "hx"},
        {"type": "phoneme", "token": "zh", "start": 0.45, "end": 0.5, "confidence": 0.87},
        {"type": "phoneme", "token": "ao", "start": 0.5, "end": 0.6, "confidence": 0.91},
        {"type": "phoneme", "token": "j", "start": 0.6, "end": 0.7, "confidence": 0.89},
        {"type": "event", "token": "[laugh]", "start": 0.85, "end": 0.95, "confidence": 0.75, "event_type": "laugh"},
    ]


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "tts: marks tests that require TTS model"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试项，自动添加标记"""
    for item in items:
        # 如果测试函数名包含 chattts 或 generate，自动添加 tts 标记
        if "chattts" in item.nodeid or "generate" in item.nodeid:
            item.add_marker(pytest.mark.tts)
            item.add_marker(pytest.mark.slow)
        # 如果测试函数名包含 integration，自动添加 integration 标记
        if "integration" in item.nodeid or "pipeline" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
