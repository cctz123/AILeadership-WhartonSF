import cv2
from deepface import DeepFace
import numpy as np
import torch
import pyaudio
import time
from threading import Thread
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

# ---------- Audio Setup ----------
model_name = "superb/wav2vec2-large-superb-er"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
audio_model = AutoModelForAudioClassification.from_pretrained(model_name)

RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 3
FORMAT = pyaudio.paInt16
audio_label = "..."  
audio_conf = 0.0

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# ---------- Audio Emotion Thread ----------
def audio_emotion_loop():
    global audio_label, audio_conf
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        audio_data = np.concatenate(frames).astype(np.float32) / 32768.0

        inputs = processor(torch.tensor(audio_data), sampling_rate=RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = audio_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            predicted_label = torch.argmax(scores).item()
            audio_label = audio_model.config.id2label[predicted_label]
            audio_conf = scores[predicted_label].item()

# Start audio thread
Thread(target=audio_emotion_loop, daemon=True).start()

# ---------- Camera Loop ----------
cap = cv2.VideoCapture(0)

# Reduce camera resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 430)

# DeepFace is slow, so only run every X frames
frame_skip = 5
frame_count = 0
face_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run DeepFace every `frame_skip` frames to improve FPS
    if frame_count % frame_skip == 0:
        try:
            face_results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            print("Face analysis error:", e)
            face_results = []

    # Draw results from last valid DeepFace inference
    for face in face_results:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        emotion = face['dominant_emotion']
        confidence = face['emotion'][emotion]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show audio tone label
    if audio_label:
        tone_text = f"Voice: {audio_label} ({audio_conf*100:.1f}%)"
        cv2.putText(frame, tone_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("🎥 Multimodal Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
