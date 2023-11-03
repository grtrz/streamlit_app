import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

df = pd.read_csv('qa.csv')
questions = df['Вопрос']
answers = df['Ответ']

embeddings = np.load('embeddings_myschool.npy')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

user_input = st.text_input('Введите вопрос:')
emb = model.encode(user_input)
cosine_scores = np.dot(emb, embeddings.T) / (np.linalg.norm(emb) * np.linalg.norm(embeddings, axis=1))
top_k = 3
top_k_idx = np.argsort(cosine_scores)[::-1][:top_k]
selected_answers = []
if user_input != '':
    for idx, i in zip(top_k_idx, range(1, top_k+1)):
        st.write(f'{i}. Вопрос: {questions[idx]}')
        selected_answers.append(answers[idx])
    idx = st.selectbox('Выберите номер вопроса: ', range(1, top_k+1)) - 1
    st.write(f'Ответ: {selected_answers[idx]}')
