import torch
import pyaudio
import numpy as np
import soundfile as sf
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

model_name = "superb/wav2vec2-large-superb-er"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)


# Audio config
RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 3
FORMAT = pyaudio.paInt16

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("🎙️  Listening for tone... Press Ctrl+C to stop.")

try:
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        audio_data = np.concatenate(frames).astype(np.float32) / 32768.0  # normalize

        # Process and predict
        inputs = processor(torch.tensor(audio_data), sampling_rate=RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            predicted_label = torch.argmax(scores).item()
            confidence = scores[predicted_label].item()

        label = model.config.id2label[predicted_label]
        print(f"🗣️  Detected Tone: {label} ({confidence * 100:.1f}%)")

except KeyboardInterrupt:
    print("\n🎤 Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
