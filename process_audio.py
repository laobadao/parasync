#!/usr/bin/env python3
"""
ParaSync Audio Processing Pipeline
完整音频处理流程：ASR -> 对齐 -> 导出
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from aligner.asr_recognizer import ASRRecognizer
from aligner.phoneme_aligner import PhonemeAligner
from aligner.nonverbal_detector import NonverbalEventDetector, merge_alignments
from aligner.textgrid_exporter import TextGridExporter
import textgrid
import soundfile as sf


def process_audio(
    audio_path: str,
    output_dir: str = ".",
    language: str = "zh",
    use_asr: bool = True,
    provided_text: str = None,
    verbose: bool = True
):
    """
    处理音频文件：ASR识别 -> 音素对齐 -> 事件检测 -> 导出TextGrid

    Args:
        audio_path: 输入音频路径 (wav, m4a, mp3)
        output_dir: 输出目录
        language: 语言代码 (zh, en)
        use_asr: 是否使用ASR识别（False则使用提供的文本）
        provided_text: 提供的参考文本（当use_asr=False时使用）
        verbose: 是否打印详细信息

    Returns:
        dict: 处理结果
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查音频格式并转换
    wav_path = audio_path
    if audio_path.suffix.lower() in ['.m4a', '.mp3']:
        if verbose:
            print(f"🔄 转换音频格式: {audio_path.suffix} -> .wav")
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        wav_path = output_dir / f"{audio_path.stem}_16k.wav"
        torchaudio.save(str(wav_path), waveform, 16000)
        if verbose:
            print(f"   保存为: {wav_path}")

    # 获取音频时长
    info = sf.info(str(wav_path))
    duration = info.duration

    if verbose:
        print(f"\n🎵 音频: {audio_path.name}")
        print(f"   时长: {duration:.2f}s")
        print(f"   采样率: {info.samplerate}Hz")

    # Step 1: ASR 识别（如果需要）
    if use_asr:
        if verbose:
            print(f"\n🔍 Step 1/4: ASR 语音识别...")
        recognizer = ASRRecognizer(
            model_name="openai/whisper-base",
            language=language
        )
        asr_result = recognizer.recognize(str(wav_path))
        text = asr_result.text
        if verbose:
            print(f"   识别结果: \"{text}\"")
            print(f"   置信度: {asr_result.confidence:.2f}")
    else:
        text = provided_text
        if verbose:
            print(f"\n📝 使用提供文本: \"{text}\"")

    if not text:
        print("❌ 错误: 无法获取文本")
        return None

    # Step 2: 音素对齐
    if verbose:
        print(f"\n🎯 Step 2/4: 音素对齐...")
    aligner = PhonemeAligner(lang=language)
    phonemes = aligner.align_text(str(wav_path), text)

    if verbose:
        print(f"   对齐完成: {len(phonemes)} 个音素段")
        for seg in phonemes:
            print(f"   {seg.token}: {seg.start_time:.3f}s - {seg.end_time:.3f}s")

    # Step 3: 事件检测
    if verbose:
        print(f"\n👂 Step 3/4: 非语言事件检测...")
    detector = NonverbalEventDetector(
        method="heuristic",
        sample_rate=16000
    )
    events = detector.detect(str(wav_path))

    if verbose:
        print(f"   检测到 {len(events)} 个事件:")
        for evt in events:
            evt_type = "🤫静音" if evt.event_type.value == "sil" else \
                      "💨呼吸" if evt.event_type.value == "hx" else \
                      "😄笑声" if evt.event_type.value == "laugh" else f"[{evt.event_type.value}]"
            print(f"   {evt_type}: {evt.start_time:.3f}s - {evt.end_time:.3f}s (置信度: {evt.confidence:.2f})")

    # Step 4: 融合并导出
    if verbose:
        print(f"\n🔄 Step 4/4: 融合并导出...")
    merged = merge_alignments(phonemes, events, preserve_phonemes=True)

    if verbose:
        phoneme_count = sum(1 for m in merged if m['type'] == 'phoneme')
        event_count = sum(1 for m in merged if m['type'] == 'event')
        print(f"   融合完成: {len(merged)} 个片段")
        print(f"   - 音素: {phoneme_count}")
        print(f"   - 事件: {event_count}")

    # 导出 TextGrid
    output_path = output_dir / f"{audio_path.stem}.TextGrid"
    exporter = TextGridExporter()

    # 计算实际时长（确保包含所有片段）
    actual_duration = max(
        duration,
        max((s['end'] for s in merged), default=duration)
    )

    # 手动创建更完整的 TextGrid
    tg = exporter.create_textgrid(actual_duration)

    # 分离事件和音素，并确保时间在范围内
    events_list = [s for s in merged if s['type'] == 'event']
    phonemes_list = [s for s in merged if s['type'] == 'phoneme']

    # 添加事件层（裁剪超出时长的部分）
    exporter.add_event_tier(tg, [
        {
            'start': min(e['start'], actual_duration - 0.001),
            'end': min(e['end'], actual_duration),
            'label': e['token']
        }
        for e in events_list
    ])

    # 添加词层
    word_tier = tg.getFirst('Word')
    for p in phonemes_list:
        if p['token'] not in ['|', '<pad>'] and p['end'] - p['start'] > 0.01:
            word_tier.addInterval(textgrid.Interval(
                minTime=p['start'],
                maxTime=p['end'],
                mark=p['token']
            ))

    tg.write(str(output_path))

    if verbose:
        print(f"\n✅ 导出成功: {output_path}")

    return {
        'audio_path': str(wav_path),
        'text': text,
        'phonemes': phonemes,
        'events': events,
        'merged': merged,
        'output_path': str(output_path),
        'duration': duration
    }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='ParaSync 音频处理工具')
    parser.add_argument('audio', help='输入音频文件路径')
    parser.add_argument('-o', '--output', default='.', help='输出目录')
    parser.add_argument('-l', '--language', default='zh', help='语言代码 (zh, en)')
    parser.add_argument('-t', '--text', help='提供参考文本（跳过ASR）')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    result = process_audio(
        audio_path=args.audio,
        output_dir=args.output,
        language=args.language,
        use_asr=(args.text is None),
        provided_text=args.text,
        verbose=not args.quiet
    )

    if result:
        print(f"\n📊 处理完成!")
        print(f"   音频: {result['audio_path']}")
        print(f"   文本: {result['text']}")
        print(f"   时长: {result['duration']:.2f}s")
        print(f"   输出: {result['output_path']}")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
