import torch
from NatureLM.models import NatureLM
import torchaudio
import time

print("Loading model...")
start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
model = model.to(device).eval()

print(f"Model loaded in {time.time() - start:.1f}s")

# 테스트 오디오 파일
audio_path = "/fs/dss/home/gaad2403/bachelor_thesis/anuraset/anuraset/audio/INCT17/INCT17_20191113_040000_0_3.wav"

print("Extracting embedding...")
start = time.time()

waveform, sr = torchaudio.load(audio_path)
print(f"Original audio shape: {waveform.shape}, Sample rate: {sr}")

# stereo/multi-channel이면 mono로 변환
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# NatureLM inference code 기준 sample rate는 16kHz
target_sr = 16000
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resampler(waveform)
    sr = target_sr

# shape: [batch, samples]
# torchaudio.load 결과가 mono면 이미 [1, samples]라 batch처럼 사용 가능
waveform = waveform.to(device)

print(f"Processed audio shape: {waveform.shape}, Sample rate: {sr}")

with torch.no_grad():
    audio_embeds, audio_atts = model.encode_audio(waveform)

print(f"Audio embeds shape: {audio_embeds.shape}")
print(f"Audio attention shape: {audio_atts.shape}")

# 파일 하나당 하나의 embedding vector가 필요하면 time/window 차원 평균
mask = audio_atts.bool().unsqueeze(-1)
file_embedding = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

print(f"File-level embedding shape: {file_embedding.shape}")
print(f"Embedding extracted in {time.time() - start:.1f}s")
print("SUCCESS!")