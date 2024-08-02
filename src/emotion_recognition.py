import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionRecognizer:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '../models/emotion_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict_emotion(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        prediction = self.model.predict(face)
        return self.emotions[np.argmax(prediction)]
