import cv2
from deepface import DeepFace
import numpy as np

# Load webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion']
            confidence = face['emotion'][emotion]

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display emotion and confidence
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    except Exception as e:
        print("Error:", e)

    # Show frame
    cv2.imshow("Live Emotion Detection", frame)

    # Break on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
