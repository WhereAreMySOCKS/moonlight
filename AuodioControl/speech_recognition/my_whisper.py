import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


class MyWhisper:
      def __init__(self, device, model_path):
            # 加载预训练模型和处理器
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device).eval()
            self.processor = WhisperProcessor.from_pretrained(model_path)
            # 设置强制解码器，解码为中文
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="zh",
                                                                                         task="transcribe")

      def read_wav(self, filepath):
            # 加载音频文件
            audio, sample_rate = librosa.load(filepath, sr=None)
            if sample_rate != 16000:
                  # 重新采样到16000Hz
                  audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            # 将音频数据转换为模型的输入特征
            return audio

      @torch.no_grad()
      def recognize_audio(self, audio):
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            # 使用模型生成预测
            predicted_ids = self.model.generate(input_features.to(self.model.device))
            # 将预测的ID序列转换为文本并返回
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

# 使用示例
# # 设定设备为CPU或CUDA
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # 创建MyWhisper类的实例
# whisper = MyWhisper(device)
#
# # 读取WAV文件并获取输入特征
# wav_file_path = '_1.wav'  # 替换为你的WAV文件路径
# input_features = whisper.read_wav(wav_file_path)
#
# # 使用模型生成文本转录
# transcription = whisper.run(input_features)
#
# # 打印转录结果
# print(transcription)
