# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render
from .models import EmotionAnalysis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from transformers import pipeline
import os
from django.conf import settings
import numpy as np
import matplotlib
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


matplotlib.use('Agg')

def home(request):
    return render(request, 'analyzer/home.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        print(f"Attempting login for user: {username}")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            print("Login successful")
            messages.success(request, 'Successfully logged in!')
            return redirect('home')
        else:
            print("Login failed: invalid username or password")
            messages.error(request, 'Invalid username or password')
            return redirect('home')
    return redirect('home')

def user_register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return redirect('home')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
            return redirect('home')
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            messages.success(request, 'Registration successful! You can now log in.')
            return redirect('home')
    return render(request, 'analyzer/home.html')

def user_logout(request):
    logout(request)
    messages.success(request, 'Successfully logged out!')
    return redirect('home')


# Assuming the model files are directly in the 'predictivemodel' directory
model_path = 'predictivemodel'
model = RobertaForSequenceClassification.from_pretrained(model_path)
emotion_analyzer = pipeline('text-classification', model=model_path, tokenizer=model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model.eval()

def home(request):
    return render(request, 'analyzer/home.html')

def analyze(request):
    if request.method == 'POST':
        text = request.POST['text']
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        probs = probabilities.detach().numpy()[0]

        # Map the probabilities to the emotion labels
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        emotion_scores = dict(zip(labels, probs))

        # Save to the database
        EmotionAnalysis.objects.create(
            text=text,
            anger=emotion_scores.get('anger', 0.0),
            disgust=emotion_scores.get('disgust', 0.0),
            fear=emotion_scores.get('fear', 0.0),
            joy=emotion_scores.get('joy', 0.0),
            neutral=emotion_scores.get('neutral', 0.0),
            sadness=emotion_scores.get('sadness', 0.0),
            surprise=emotion_scores.get('surprise', 0.0)
        )

        return render(request, 'analyzer/results.html', {'emotion_scores': emotion_scores, 'text': text})
    return render(request, 'analyzer/home.html')    



def generate_pie_chart(data):
    labels = [key for key, value in data.items() if value > 0]
    sizes = [value for value in data.values() if value > 0]
    if not sizes:  # if sizes list is empty, add a dummy value
        labels = ['No data']
        sizes = [1]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def generate_bar_chart(data):
    labels = list(data.keys())
    sizes = list(data.values())
    fig, ax = plt.subplots()
    ax.bar(labels, sizes)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64


def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def visualizations(request):
    analyses = EmotionAnalysis.objects.all()
    emotion_data = {
        'anger': 0.0,
        'disgust': 0.0,
        'fear': 0.0,
        'joy': 0.0,
        'neutral': 0.0,
        'sadness': 0.0,
        'surprise': 0.0
    }
    text_data = ""
    for analysis in analyses:
        emotion_data['anger'] += analysis.anger
        emotion_data['disgust'] += analysis.disgust
        emotion_data['fear'] += analysis.fear
        emotion_data['joy'] += analysis.joy
        emotion_data['neutral'] += analysis.neutral
        emotion_data['sadness'] += analysis.sadness
        emotion_data['surprise'] += analysis.surprise
        text_data += analysis.text + " "

    # Replace NaN values with 0.0
    for emotion, score in emotion_data.items():
        if np.isnan(score):
            emotion_data[emotion] = 0.0

    pie_chart = generate_pie_chart(emotion_data)
    bar_chart = generate_bar_chart(emotion_data)

    # Check if text_data is empty before generating word cloud
    if text_data:
        word_cloud = generate_word_cloud(text_data)
    else:
        word_cloud = None

    return render(request, 'analyzer/visualizations.html', {
        'pie_chart': pie_chart,
        'bar_chart': bar_chart,
        'word_cloud': word_cloud
    })
