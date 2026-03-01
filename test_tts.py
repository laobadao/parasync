import torch
import ChatTTS
import soundfile as sf

chat = ChatTTS.Chat()
chat.load(compile=False)

text = "我是赵君君，[uv_break][laugh]。"

wavs = chat.infer([text])

sf.write("output.wav", wavs[0].squeeze(), 24000)
print("✅ 语音已保存到 output.wav")
