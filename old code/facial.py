import cv2
import torch
import numpy as np
from torchvision import transforms
from emo_affectnet import ResNet50  # adjust import based on repo structure

# === 1. Load Pre-trained Model ===
num_classes = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
model = ResNet50(num_classes=num_classes, channels=3)
model.load_state_dict(torch.load("FER_static_ResNet50_AffectNet.pt", map_location="cpu"))
model.eval()

# === 2. Preprocessing Pipeline ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# === 3. Real-time Webcam Loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB and preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(img).unsqueeze(0)  # shape [1,3,224,224]

    # Run inference
    with torch.no_grad():
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).item()
    emotion = emotion_labels[pred]

    # Display
    cv2.putText(frame, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)
    cv2.imshow("AffectNet Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
