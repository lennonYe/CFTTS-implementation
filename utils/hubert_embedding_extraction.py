import torch
import os
import librosa
from transformers import HubertModel  
  
# 加载 HuBERT 模型  
model_name = "facebook/hubert-large-ll60k"  
model = HubertModel.from_pretrained(model_name)  
model.eval()  # 设置模型为评估模式  
  

train_dir = '/root/autodl-tmp/VoiceFlow-TTS/data/ljspeech/train/utts.list'
train_wavs = []
with open(train_dir,'r') as f:
    for line in f:
        cur = line.strip()
        train_wavs.append(cur)
print(train_wavs)

wav_path = '/root/autodl-tmp/datasets/LJSpeech-1.1/wav_16k'

for dir in train_wavs:
    cur_dir = wav_path + '/' + dir + '.wav'
    save_dir = '/root/autodl-tmp/VoiceFlow-TTS/data/ljspeech/train_hubert' + '/' + dir + '.pt'
    waveform, sample_rate = librosa.load(cur_dir,sr=16000)  
    waveform = torch.from_numpy(waveform)
    waveform = waveform.unsqueeze(0)
    with torch.no_grad():  
    # 将音频数据传入模型，获取输出  
        outputs = model(waveform)  
    last_hidden_states = outputs.last_hidden_state  
    torch.save(last_hidden_states,save_dir)
  


