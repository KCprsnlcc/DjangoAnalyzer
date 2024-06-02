from django.shortcuts import render
from transformers import pipeline

# Assuming the model files are directly in the 'predictivemodel' directory
model_path = 'predictivemodel'

emotion_analyzer = pipeline('text-classification', model=model_path, tokenizer=model_path)

def home(request):
    return render(request, 'analyzer/home.html')

def analyze(request):
    if request.method == 'POST':
        text = request.POST['text']
        results = emotion_analyzer(text)
        return render(request, 'analyzer/results.html', {'results': results, 'text': text})
    return render(request, 'analyzer/home.html')
