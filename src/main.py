import sys
import os
import cv2
import webbrowser
import time
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from music_recommendation import MusicRecommender
from mood_tracking import MoodTracker

def main():
    print("Initializing Face Detector")
    face_detector = FaceDetector()
    print("Initializing Emotion Recognizer")
    emotion_recognizer = EmotionRecognizer()
    print("Initializing Music Recommender")
    music_recommender = MusicRecommender()
    print("Initializing Mood Tracker")
    mood_tracker = MoodTracker()

    cap = cv2.VideoCapture(0)
    last_emotion = None
    last_detection_time = 0
    detection_interval = 300

    devices = music_recommender.get_devices()
    device_map = {i + 1: device for i, device in enumerate(devices)}
    for idx, device in device_map.items():
        print(f"{idx}: {device['name']} ({device['type']})")

    device_choice = int(input("Choose a device for playback: "))
    if device_choice in device_map:
        selected_device = device_map[device_choice]
        print(f"Selected device: {selected_device['name']} ({selected_device['type']})")
    else:
        selected_device = None
        print("Invalid choice. Defaulting to web browser playback.")

    while True:
        ret, frame = cap.read()
        faces, landmarks = face_detector.detect_faces(frame)

        for face, landmark in zip(faces, landmarks):
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            face_img = frame[y1:y2, x1:x2]
            current_time = time.time()

            if current_time - last_detection_time >= detection_interval:
                emotion = emotion_recognizer.predict_emotion(face_img)
                mood_tracker.update_mood_history(emotion)

                if emotion != last_emotion:
                    last_emotion = emotion
                    last_detection_time = current_time
                    track_name, track_uri = music_recommender.recommend_music(emotion)
                    if track_name:
                        if selected_device:
                            music_recommender.start_playback(selected_device['id'], track_uri)
                        else:
                            open_spotify_track(track_uri)
                        cv2.putText(frame, track_name, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, last_emotion if last_emotion else emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def open_spotify_track(track_uri):
    try:
        webbrowser.open(f"https://open.spotify.com/track/{track_uri.split(':')[-1]}")
    except Exception as e:
        print(f"Error opening song: {e}")

if __name__ == "__main__":
    main()
