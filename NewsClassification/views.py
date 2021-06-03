from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle



def index(request):
    return render(request, 'index.html')


def predict(request):
    # Get the text
    category_list = ['business', 'entertainment', 'politics', 'sport', 'tech']

    # Check checkbox values
    Classify = request.POST.get('text')
    docs_new = [Classify]
    # LOAD MODEL
    loaded_tfidf = pickle.load(open("tfidf.pkl", "rb"))
    loaded_model = pickle.load(open("svc_model.pkl", "rb"))

    X_new_tfidf = loaded_tfidf.transform(docs_new)
    predicted = loaded_model.predict(X_new_tfidf)
    output = category_list[predicted[0]]
    params = {'predicted_class': output}
    return render(request, 'predict.html', params)

def about(request):
    return render(request, 'about.html')