from collections import deque


class MoodTracker:
    def __init__(self, maxlen=100):
        self.mood_history = deque(maxlen=maxlen)

    def update_mood_history(self, emotion):
        self.mood_history.append(emotion)

    def get_mood_trends(self):
        mood_counts = {}
        for mood in self.mood_history:
            if mood in mood_counts:
                mood_counts[mood] += 1
            else:
                mood_counts[mood] = 1
        return mood_counts

    def get_most_common_mood(self):
        trends = self.get_mood_trends()
        most_common_mood = max(trends, key=trends.get)
        return most_common_mood, trends[most_common_mood]

    def reset_mood_history(self):
        self.mood_history.clear()
