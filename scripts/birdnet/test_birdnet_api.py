#!/usr/bin/env python3
"""
BirdNET model.encode() API 테스트
AcousticFileEncodingResult 객체 구조 확인
"""
import birdnet
from pathlib import Path

BASE_DIR = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR = BASE_DIR / "ESC-50-master/audio"

# 모델 로드
print("Loading BirdNET model...")
model = birdnet.load("acoustic", "2.4", "tf")

# 테스트 오디오 파일 경로
test_file = str(AUDIO_DIR / "1-100032-A-0.wav")
print(f"Testing with: {test_file}")

# 단일 파일 encode 시도
print("\n=== Single file encode ===")
result = model.encode([test_file])
print(f"Type of result: {type(result)}")
print(f"Result: {result}")

# 속성/메서드 탐색
print(f"\nAttributes/Methods of result:")
for attr in dir(result):
    if not attr.startswith("_"):
        print(f"  - {attr}")

# 각 속성 확인
print(f"\n=== Trying to access embeddings ===")
try:
    print(f"result.embeddings: {result.embeddings}")
except AttributeError as e:
    print(f"✗ result.embeddings failed: {e}")

try:
    print(f"result.get_embeddings(): {result.get_embeddings()}")
except AttributeError as e:
    print(f"✗ result.get_embeddings() failed: {e}")

try:
    print(f"result.data: {result.data}")
except AttributeError as e:
    print(f"✗ result.data failed: {e}")

try:
    print(f"result.output: {result.output}")
except AttributeError as e:
    print(f"✗ result.output failed: {e}")

# numpy 변환 시도
print(f"\n=== Trying numpy conversions ===")
try:
    import numpy as np
    arr = np.array(result)
    print(f"np.array(result) shape: {arr.shape}")
    print(f"np.array(result) dtype: {arr.dtype}")
except Exception as e:
    print(f"✗ np.array() failed: {e}")

# 길이 체크
print(f"\n=== Length check ===")
try:
    print(f"len(result): {len(result)}")
except TypeError as e:
    print(f"✗ len() failed: {e}")

print("\n=== End of test ===")