import pandas as pd  
import random
import re
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score
import gensim.downloader as api

# Télécharger le modèle Word2Vec pré-entraîné
w2v_model = api.load("word2vec-google-news-300")

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation, les symboles et les caractères spéciaux (à l'exception des hashtags et des mentions)
    text = re.sub(r'[^\w\s#@]', '', text)
    # Jetonisation
    tokens = word_tokenize(text)
    # Supprimer les mots vides
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Load the training data
train_data = pd.read_excel('/content/drive/MyDrive/dataset/train_modifier.xlsx')

# Load the test data
test_data = pd.read_excel('/content/drive/MyDrive/dataset/testmodifier.xlsx')

# Create DataFrames for training and testing
df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Obtenir des paires du même utilisateur avec prétraitement du texte
same_user_pairs_train = [(preprocess_text(row['text']), preprocess_text(row2['text']), 1) for _, row in df_train.iterrows() for _, row2 in df_train.iterrows() if row['user'] == row2['user']]

# Obtenir des paires d'utilisateurs différents avec prétraitement du texte
user_pairs_train = [(preprocess_text(row['text']), preprocess_text(row2['text']), 0) for _, row in df_train.iterrows() for _, row2 in df_train.iterrows() if row['user'] != row2['user']]

# Échantillonner un nombre égal de paires d'utilisateurs différents
sampled_user_pairs_train = random.sample(user_pairs_train, min(len(same_user_pairs_train), len(user_pairs_train)))

# Combiner les paires et les mélanger
sampled_pairs_train = same_user_pairs_train + sampled_user_pairs_train
random.shuffle(sampled_pairs_train)

# Convertir en DataFrame
df_pairs_train = pd.DataFrame(sampled_pairs_train, columns=['text1', 'text2', 'similarity_label'])

# Calculer les embeddings pour chaque texte en utilisant Word2Vec
def get_text_embeddings(text):
    text_embedding = np.zeros(w2v_model.vector_size)
    word_count = 0
    for word in text:
        if word in w2v_model:
            text_embedding += w2v_model[word]
            word_count += 1
    if word_count != 0:
        text_embedding /= word_count
    return text_embedding

# Ajouter une colonne pour les embeddings de texte 1
df_pairs_train['text1_embedding'] = df_pairs_train['text1'].apply(get_text_embeddings)

# Ajouter une colonne pour les embeddings de texte 2
df_pairs_train['text2_embedding'] = df_pairs_train['text2'].apply(get_text_embeddings)

# Calculer la similarité entre les embeddings de texte 1 et 2 en utilisant la distance de Manhattan
df_pairs_train['similarity_score'] = df_pairs_train.apply(lambda row: np.abs(row['text1_embedding'] - row['text2_embedding']).sum(), axis=1)

# Utiliser le modèle pour prédire les étiquettes de similarité
predicted_similarity_labels_train = [1 if score >= 0.5 else 0 for score in df_pairs_train['similarity_score']]

# Calculer les métriques d'évaluation pour les données d'entraînement
precision_train = precision_score(df_pairs_train['similarity_label'], predicted_similarity_labels_train)
recall_train = recall_score(df_pairs_train['similarity_label'], predicted_similarity_labels_train)
f1_train = f1_score(df_pairs_train['similarity_label'], predicted_similarity_labels_train)

# Afficher les métriques d'évaluation pour les données d'entraînement
print("Training Precision:", precision_train)
print("Training Recall:", recall_train)
print("Training F1 Score:", f1_train)

# Obtenir des paires du même utilisateur avec prétraitement du texte pour les données de test
same_user_pairs_test = [(preprocess_text(row['text']), preprocess_text(row2['text']), 1) for _, row in df_test.iterrows() for _, row2 in df_test.iterrows() if row['user'] == row2['user']]

# Obtenir des paires d'utilisateurs différents avec prétraitement du texte pour les données de test
user_pairs_test = [(preprocess_text(row['text']), preprocess_text(row2['text']), 0) for _, row in df_test.iterrows() for _, row2 in df_test.iterrows() if row['user'] != row2['user']]

# Échantillonner un nombre égal de paires d'utilisateurs différents pour les données de test
sampled_user_pairs_test = random.sample(user_pairs_test, min(len(same_user_pairs_test), len(user_pairs_test)))

# Combiner les paires et les mélanger pour les données de test
sampled_pairs_test = same_user_pairs_test + sampled_user_pairs_test
random.shuffle(sampled_pairs_test)

# Convertir en DataFrame pour les données de test
df_pairs_test = pd.DataFrame(sampled_pairs_test, columns=['text1', 'text2', 'similarity_label'])

# Calculer les embeddings pour chaque texte en utilisant Word2Vec pour les données de test
df_pairs_test['text1_embedding'] = df_pairs_test['text1'].apply(get_text_embeddings)
df_pairs_test['text2_embedding'] = df_pairs_test['text2'].apply(get_text_embeddings)

# Calculer la similarité entre les embeddings de texte 1 et 2 en utilisant la distance de Manhattan pour les données de test
df_pairs_test['similarity_score'] = df_pairs_test.apply(lambda row: np.abs(row['text1_embedding'] - row['text2_embedding']).sum(), axis=1)

# Utiliser le modèle pour prédire les étiquettes de similarité pour les données de test
predicted_similarity_labels_test = [1 if score >= 0.5 else 0 for score in df_pairs_test['similarity_score']]

# Calculer les métriques d'évaluation pour les données de test
precision_test = precision_score(df_pairs_test['similarity_label'], predicted_similarity_labels_test)
recall_test = recall_score(df_pairs_test['similarity_label'], predicted_similarity_labels_test)
f1_test = f1_score(df_pairs_test['similarity_label'], predicted_similarity_labels_test)

# Afficher les métriques d'évaluation pour les données de test
print("\nTesting Precision:", precision_test)
print("Testing Recall:", recall_test)
print("Testing F1 Score:", f1_test)
