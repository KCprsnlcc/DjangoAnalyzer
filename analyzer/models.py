from django.db import models

class EmotionAnalysis(models.Model):
    text = models.TextField()
    anger = models.FloatField()
    disgust = models.FloatField()
    fear = models.FloatField()
    joy = models.FloatField()
    neutral = models.FloatField()
    sadness = models.FloatField()
    surprise = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Add latitude and longitude fields
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.text
