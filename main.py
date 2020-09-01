import numpy as np
import pandas as pd
import random
import torch
from fastai import *
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

data = pd.read_csv('sample.txt',sep=" - ")

folder = 'D:/danklearning/project_image_captioning/images'
images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)
#cv2.imshow('image', images[0])

def prep(data):
    corpus = data.lower().split("\n")    
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

def lstmmodel(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)

def g_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
x = path.rfind('/')
img_name_temp = path[x+1:]
img_name = img_name_temp[:len(img_name_temp)-4]
img_name = img_name.replace('-', ' ')

df = data[data['label'].str.contains(img_name)]
df_size = len(df.index)
sel_row = random.randint(1, df_size)
value = df.iloc[sel_row]['caption']
print ("\nCaption:  ",value) 