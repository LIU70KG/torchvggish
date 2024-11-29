dependencies = ['torch', 'numpy', 'resampy', 'soundfile']
import os
from torchvggish.vggish import VGGish
import pickle
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}


def vggish(**kwargs):
    model = VGGish(urls=model_urls, **kwargs)
    return model


if __name__ == '__main__':

    # # 测试
    # filename = '300_AUDIO.wav'
    # new_wav_file = '300_AUDIO_new.wav'
    # filename = 'test.wav'
    # new_wav_file = 'test_new.wav'
    # # filename = 'Q1.wav'
    # # new_wav_file = 'Q1_new.wav'
    # audio_data, sample_rate = librosa.load(filename, sr=None, mono=False)
    # # 对音频数据进行处理（这里以截断为例）
    # # 确保处理后的数据与原始数据的通道数一致
    # if audio_data.ndim == 1:  # 单通道音频
    #     audio_data_new = audio_data[:len(audio_data) // 2]
    # else:  # 多通道音频
    #     audio_data_new = audio_data[:, :audio_data.shape[1] // 2]
    #
    # sf.write(new_wav_file, audio_data_new.T if audio_data_new.ndim > 1 else audio_data_new, sample_rate)
    # # 读取原始和处理后的 WAV 文件
    # audio_data_sf, sample_rate_sf = sf.read(filename, dtype='int16')
    # audio_data_new_read_sf, sample_rate_new = sf.read(new_wav_file, dtype='int16')
    #
    # # 如果原始音频是多通道，比较时也需要注意通道数
    # if audio_data_sf.ndim == 1:  # 单通道音频
    #     is_equal = np.array_equal(audio_data_sf[:len(audio_data_sf) // 2], audio_data_new_read_sf)
    # else:  # 多通道音频
    #     is_equal = np.array_equal(audio_data_sf[:audio_data_sf.shape[0] // 2, :], audio_data_new_read_sf)
    # # 打印结果
    # print(f'Are the audio data equal? {is_equal}')
    #
    # 使用模型提取音频特征
    # model = VGGish(urls=model_urls)
    # filename = '300_AUDIO.wav'
    # audio_data, sample_rate = librosa.load(filename, sr=None, mono=False)
    # features = model.forward(filename)
    # print("输出特征维度：", features.shape)
    # if audio_data.ndim == 1:  # 单通道音频
    #     print("音频秒速：", audio_data.shape[0]/sample_rate)
    # else:  # 多通道音频
    #     print("音频秒速：", audio_data.shape[1] / sample_rate)

    # ---------------使用DAIC-WOZ--------------------------
    model = VGGish(urls=model_urls)

    folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph/sample_paragraph'
    sample_list = os.listdir(folder_path)
    sample_list.sort()
    sample_paragraph_all = []
    for sample in sample_list:
        print(f"处理样本：{sample}")
        sample_paragraph_list = os.listdir(os.path.join(folder_path, sample))
        sample_paragraph_list = sorted(sample_paragraph_list, key=lambda x: int(x.split('.')[0]))
        sample_paragraph = []
        for sample_sentence in sample_paragraph_list:
            # 使用模型提取音频特征
            features = model.forward(os.path.join(folder_path, sample, sample_sentence))
            # print(f"样本句子：{sample_sentence}, 输出特征维度：{features.shape}")
            features_np = features.detach().cpu().numpy()
            if features_np.shape == (128,):
                features_np = features_np.reshape(1, -1)
            sample_paragraph.append(features_np)
        sample_paragraph_all.append(sample_paragraph)

    folder_path = '/home/liu70kg/PycharmProjects/daic_woz_process-master/extracted_database_features/audio_paragraph'
    with open(os.path.join(folder_path, 'audio_paragraph_features.pickle'), 'wb') as f:
        pickle.dump(sample_paragraph_all, f)
    print(f"语音段落的特征处理完成，保存在{os.path.join(folder_path, 'audio_paragraph_features.pickle')}")


    # ---------------使用SEARCH数据集--------------------------
    # model = VGGish(urls=model_urls)
    #
    # dataSource = '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio'
    # dataOutDir = '/home/liu70kg/PycharmProjects/Depression/SEARCH/denoised_audio_feature'
    # folderName = ['sheyang', 'taizhou', 'yixing']
    #
    # for place in folderName:
    #     dataSource1 = os.path.join(dataSource, place)
    #     dataOutDir1 = os.path.join(dataOutDir, place)
    #
    #     # 音频在某文件夹内
    #     wav_list = os.listdir(dataSource1)
    #
    #     # wav_path = os.path.join(dataSource1, wav_list[936])
    #     # try:
    #     #     features = model.forward(wav_path)
    #     # except:
    #     #     continue
    #
    #
    #     for filename in tqdm(wav_list, desc='VGGish提取特征'):
    #         wav_path = os.path.join(dataSource1, filename)
    #         outputPath = os.path.join(dataOutDir1, os.path.splitext(filename)[0] + '.npy')
    #         # print(wav_path)
    #         try:
    #             features = model.forward(wav_path)
    #         except:
    #             continue
    #         # print(f"样本句子：{sample_sentence}, 输出特征维度：{features.shape}")
    #         features_np = features.detach().cpu().numpy()
    #         if features_np.shape == (128,):
    #             features_np = features_np.reshape(1, -1)
    #         np.save(outputPath, features_np)   # 保存为 .npy 文件
    #         # loaded_arr = np.load(outputPath) # 读取 .npy 文件


    # ---------------使用AVEC数据集audio_Northwind--------------------------
    # model = VGGish(urls=model_urls)
    #
    # dataSource = '/home/liu70kg/D512G/AVEC2014/audio_Northwind'
    # dataOutDir = '/home/liu70kg/PycharmProjects/AVEC_process-master/audio_Northwind_feature'
    #
    # # 音频在某文件夹内
    # wav_list = os.listdir(dataSource)
    #
    #
    # for filename in tqdm(wav_list, desc='VGGish提取特征'):
    #     wav_path = os.path.join(dataSource, filename)
    #     outputPath = os.path.join(dataOutDir, os.path.splitext(filename)[0] + '.npy')
    #     # print(wav_path)
    #     try:
    #         features = model.forward(wav_path)
    #     except:
    #         continue
    #     # print(f"样本句子：{sample_sentence}, 输出特征维度：{features.shape}")
    #     features_np = features.detach().cpu().numpy()
    #     if features_np.shape == (128,):
    #         features_np = features_np.reshape(1, -1)
    #     np.save(outputPath, features_np)   # 保存为 .npy 文件
    #     # loaded_arr = np.load(outputPath) # 读取 .npy 文件

    # ---------------使用AVEC数据集audio_Freeform--------------------------
    # model = VGGish(urls=model_urls)
    #
    # dataSource = '/home/liu70kg/D512G/AVEC2014/audio_Freeform'
    # dataOutDir = '/home/liu70kg/PycharmProjects/AVEC_process-master/audio_Freeform_feature'
    #
    # # 音频在某文件夹内
    # wav_list = os.listdir(dataSource)
    #
    #
    # for filename in tqdm(wav_list, desc='VGGish提取特征'):
    #     wav_path = os.path.join(dataSource, filename)
    #     outputPath = os.path.join(dataOutDir, os.path.splitext(filename)[0] + '.npy')
    #     # print(wav_path)
    #     try:
    #         features = model.forward(wav_path)
    #     except:
    #         continue
    #     # print(f"样本句子：{sample_sentence}, 输出特征维度：{features.shape}")
    #     features_np = features.detach().cpu().numpy()
    #     if features_np.shape == (128,):
    #         features_np = features_np.reshape(1, -1)
    #     np.save(outputPath, features_np)   # 保存为 .npy 文件
    #     # loaded_arr = np.load(outputPath) # 读取 .npy 文件
