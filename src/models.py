# models.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score
import config

def extract_tfidf_features(df):
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES, 
        ngram_range=(1, 2), 
        sublinear_tf=True, 
        min_df=2, 
        stop_words='english'
    )
    X = tfidf.fit_transform(df['clean_body'])
    return X, tfidf

def train_classifiers(X, df):
    y_multi = np.column_stack([df['priority_label'], df['type_label'], df['queue_label']])
    X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.30, random_state=42)

    models = {
        'Logistic Regression': MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        'Linear SVC': MultiOutputClassifier(CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, max_iter=2000))),
        'Random Forest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    }

    results = []
    trained_instances = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_instances[name] = model
        
        p_acc = accuracy_score(y_test[:,0], preds[:,0])
        t_acc = accuracy_score(y_test[:,1], preds[:,1])
        q_acc = accuracy_score(y_test[:,2], preds[:,2])
        ovf1 = (f1_score(y_test[:,0], preds[:,0], average='macro') + 
                f1_score(y_test[:,1], preds[:,1], average='macro') + 
                f1_score(y_test[:,2], preds[:,2], average='macro')) / 3
        
        results.append({'Model': name, 'Priority_Acc': p_acc, 'Type_Acc': t_acc, 'Queue_Acc': q_acc, 'Overall_F1': ovf1})

    sorted_results = sorted(results, key=lambda x: x['Overall_F1'], reverse=True)
    best_model_name = sorted_results[0]['Model']
    return trained_instances[best_model_name], sorted_results

def build_faiss_index(df, embedder):
    print("⏳ Processing neural text embeddings via SBERT...")
    embeddings = embedder.encode(df['clean_body'].tolist(), show_progress_bar=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, embeddings