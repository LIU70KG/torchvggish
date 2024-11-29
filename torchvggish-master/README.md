依据在YouTube-8M预训练的VGGish模型，对各样本的单次回答语音提取n*128的特征，特征保存在文件audio_paragraph_features.pickle。
为什么是n*128?因为1秒的语音特征是1*128，如果单次回答是5.3秒，那特征就是5*128。

执行入口：torchvggish-master里的main.py

---


# VGGish
下载链接中的参数，并保存到“torchvggish-master”目录下。
[VGGish](https://drive.google.com/drive/folders/1nRNK8x-7i7a87naxmLTCdi7SjSDaxyqG?usp=sharing)<sup>[1]</sup>, 



```

<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

