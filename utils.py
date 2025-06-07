import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Constants
def extract_genres(json_str):
    """Extract genre names from a JSON string."""
    try:
        genres = ast.literal_eval(json_str)
        genre_names = [g['name'] for g in genres]
        return genre_names
    except:
        return []
    
def get_single_valid_genre(g_list):
    return g_list[0] if len(g_list) == 1 and g_list[0] in valid_genres else None


def count_chunks(text, token_splitter):
    """Count the number of chunks in a text using the token splitter."""
    chunks = token_splitter.split_text(text)
    return len(chunks)

def get_topic_features(X, lda_model):
    """Get topic features using LDA model."""
    return lda_model.fit_transform(X)

def evaluate_lda_model(X_bow, y, n_topics_list, n_splits=10, random_state=42):
    """Evaluate LDA model with different numbers of topics using cross-validation."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_results = {}

    for n_topics in n_topics_list:
        print(f"\nEvaluando modelo con {n_topics} tópicos...")
        
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method='batch')
        X_topics = get_topic_features(X_bow, lda_model)
        
        clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state)
        
        scores = []
        for train_idx, test_idx in kfold.split(X_topics, y):
            X_train, X_test = X_topics[train_idx], X_topics[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_prob)
            scores.append(score)
        
        mean_auc = np.mean(scores)
        auc_results[n_topics] = mean_auc
        print(f"ROC-AUC promedio para {n_topics} tópicos: {mean_auc:.4f}")
    
    return auc_results

def get_top_words_per_topic(lda_model, vocab, n_top_words=5):
    """Get the top words for each topic in the LDA model."""
    top_words_per_topic = []
    for topic in lda_model.components_:
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words_per_topic.append([vocab[i] for i in top_indices])
    return top_words_per_topic

def predict_genre_probability(sinopsis, vectorizer, lda_model, clf):
    """Predict the probability of a synopsis being a comedy."""
    X_bow = vectorizer.transform([sinopsis])
    X_topics = lda_model.transform(X_bow)
    prob = clf.predict_proba(X_topics)[:, 1][0]
    return prob 

def predict_with_slda_models(sinopsis_list, vectorizer, slda_models):
    """
    Predice la probabilidad de pertenecer a cada género usando modelos sLDA entrenados.

    Returns:
        DataFrame: filas = sinopsis, columnas = géneros, valores = probabilidad
    """
    bow = vectorizer.transform(sinopsis_list)
    pred_dict = {}

    for genre, model in slda_models.items():
        lda = model['lda']
        clf = model['clf']

        topic_dist = lda.transform(bow)
        probas = clf.predict_proba(topic_dist)[:, 1]
        pred_dict[genre] = probas

    return pd.DataFrame(pred_dict, index=[f"Sinopsis {i+1}" for i in range(len(sinopsis_list))])

def train_slda_models(df_texts, X_bow, genres, n_topics):
    """
    Entrena un modelo LDA + Regresión Logística para cada género como clasificación binaria.
    
    Args:
        df_texts: DataFrame que contiene la columna 'genre'
        X_bow: matriz sparse de CountVectorizer
        genres: lista de géneros válidos
        n_topics: número de temas para el LDA

    Returns:
        Diccionario con los modelos por género: {'GenreName': {'lda': model, 'clf': model}}
    """
    models = {}

    for genre in genres:
        print(f"Entrenando modelo para género: {genre}")

        # Codificar target binaria para ese género
        y = df_texts['genre'].apply(lambda g: 1 if g == genre else 0).values

        # LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
        X_topics = lda.fit_transform(X_bow)

        # Clasificador
        clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        clf.fit(X_topics, y)

        # Guardar modelo
        models[genre] = {'lda': lda, 'clf': clf}

    return models