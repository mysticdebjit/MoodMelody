
# MoodMelody

This project detects human emotions in real-time using facial expressions and suggests music or content based on the detected mood. The user can even choose to play the recommended music either in the web browser or on a selected Spotify device.

## Features
- Real-time emotion detection using OpenCV and dlib
- Music recommendation based on detected emotions
- Playback choice between web browser and selected Spotify device
- 5-minute interval for mood change detection to avoid continuous detection

## Technologies
- OpenCV
- dlib
- numpy
- TensorFlow/Keras
- Spotipy (Spotify API)
- python-dotenv
- webbrowser

## Setup

### Prerequisites
- Python 3.x
- Spotify Developer Account
- Spotify API credentials (Client ID, Client Secret)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/debjit-mandal/MoodMelody.git
    cd MoodMelody
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project root directory with the following content:

    ```plaintext
    SPOTIFY_CLIENT_ID=your_spotify_client_id
    SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
    SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
    ```

    Replace `your_spotify_client_id` and `your_spotify_client_secret` with your actual Spotify API credentials.

## Usage

1. **Run the project:**

    ```bash
    python src/main.py
    ```

2. **Choose a device for playback:**

    The script will list available Spotify devices. Choose a device for playback or default to web browser playback.

3. **Webcam detection and music recommendation:**

    The webcam will detect your emotion and recommend music accordingly. A new song will be played only if a new emotion is detected, with a 5-minute interval to avoid continuous detection.

## Directory Structure

MoodMelody/
│
├── src/
│ ├── face_detection.py
│ ├── emotion_recognition.py
│ ├── music_recommendation.py
│ ├── mood_tracking.py
│ ├── main.py
│ ├── train_emotion_recognition_model.py # Training script
│
├── models/
│ └── emotion_model.h5 # Trained emotion recognition model
│
├── data/
│ ├── shape_predictor_68_face_landmarks.dat # dlib face landmark model
│ └── fer2013.csv # FER2013 dataset file for training
│
├── requirements.txt
├── .gitignore
├── .env
└── README.md

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [OpenCV](https://opencv.org/)
- [dlib](http://dlib.net/)
- [TensorFlow](https://www.tensorflow.org/)
- [Spotipy](https://spotipy.readthedocs.io/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
