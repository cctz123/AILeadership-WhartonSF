import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
import librosa
import numpy as np
import cv2
import os
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

# === 1. Model Definitions ===
class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, face_feat_dim=512, audio_feat_dim=40, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(face_feat_dim + audio_feat_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, face_feats, audio_feats):
        x = torch.cat((face_feats, audio_feats), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === 2. Feature Extractors ===
def extract_face_features(video_path, resnet, transform):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        face_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (224, 224))
        face_tensor = transform(face_img)
        frames.append(face_tensor)
        success, frame = cap.read()
    cap.release()

    if not frames:
        return torch.zeros((1, 512))

    batch = torch.stack(frames)
    with torch.no_grad():
        features = resnet(batch)
    return features.mean(dim=0, keepdim=True)  # (1, 512)

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    return torch.tensor(mfcc_mean, dtype=torch.float).unsqueeze(0)  # (1, 40)

# === 3. Load Dataset ===
# dataset = load_dataset("AbstractTTS/IEMOCAP")
dataset = load_dataset("AbstractTTS/IEMOCAP", split="train[:5]")
# dataset = load_dataset("AbstractTTS/IEMOCAP", split="train", streaming=True)


label2id = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "exc": 1,  # merge happy + excited
    "fru": 4,
    "sur": 5
}

# === 4. Setup ===

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
# resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = MultimodalEmotionClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === 5. Training Loop (Toy Example on 10 Samples) ===
model.train()
# for sample in tqdm(dataset["train"][:10]):
for sample in tqdm(dataset):
    video_path = sample["video"].get("path", None)
    audio_path = sample["audio"].get("path", None)
    emotion = sample["emotion"]

    if emotion not in label2id or not video_path or not audio_path:
        continue

    face_feats = extract_face_features(video_path, resnet, transform)
    audio_feats = extract_audio_features(audio_path)
    label = torch.tensor([label2id[emotion]])

    output = model(face_feats, audio_feats)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Trained on: {video_path.split('/')[-1]}, Loss: {loss.item():.4f}")

# === 6. Save Model ===
torch.save(model.state_dict(), "multimodal_emotion_model.pth")
