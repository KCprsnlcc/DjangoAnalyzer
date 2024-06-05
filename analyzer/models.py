from django.db import models

class EmotionAnalysis(models.Model):
    text = models.TextField()
    anger = models.FloatField(default=0.0)
    disgust = models.FloatField(default=0.0)
    fear = models.FloatField(default=0.0)
    joy = models.FloatField(default=0.0)
    neutral = models.FloatField(default=0.0)
    sadness = models.FloatField(default=0.0)
    surprise = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.text[:50]} - Anger: {self.anger}, Disgust: {self.disgust}, Fear: {self.fear}, Joy: {self.joy}, Neutral: {self.neutral}, Sadness: {self.sadness}, Surprise: {self.surprise}'
