import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

class MusicRecommender:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope='user-read-private user-read-email user-read-playback-state user-modify-playback-state'
        ))

    def recommend_music(self, emotion):
        mood_map = {
            'Happy': 'happy',
            'Sad': 'sad',
            'Angry': 'angry',
            'Disgust': 'disgust',
            'Fear': 'fear',
            'Surprise': 'surprise',
            'Neutral': 'neutral'
        }
        mood = mood_map.get(emotion, 'neutral')
        results = self.sp.search(q=f'mood:{mood}', type='track', limit=10)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            track_name = track['name']
            track_uri = track['uri']
            return track_name, track_uri
        return None, None

    def get_devices(self):
        devices = self.sp.devices()
        return devices['devices']

    def start_playback(self, device_id, track_uri):
        self.sp.start_playback(device_id=device_id, uris=[track_uri])
