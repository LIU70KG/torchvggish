# 作者：刘成广
# 时间：2024/8/6 下午10:08
import torch
import torchaudio
from torchvggish import vggish
# from torchvggish import vggish, VGGishPostprocessor
import os

# 设置模型文件路径
pca_params_path = 'vggish_pca_params-970ea276.pth'
# 加载预训练的 VGGish 模型
model = vggish()
state_dict = torch.load('vggish-10086976.pth')
model.load_state_dict(state_dict)
model.eval()

folder_path = 'audio_data'
files = os.listdir(folder_path)
wav_files = [f for f in files if f.endswith('.wav')]
wav_files.sort()
# ---------------使用--------------------------
vidio_features_all = []
for filename in wav_files:
    # 使用模型提取音频特征
    waveform, sample_rate = torchaudio.load(os.path.join(folder_path, filename))
    # 如果需要，重新采样音频到 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    # 获取模型的输出
    with torch.no_grad():
        features = model(waveform)
    print(f"文件：{filename}, 输出特征维度：{features.shape}")
    vidio_features_all.append(features)

k = 1
