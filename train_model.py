#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPAM DETECTOR - Machine Learning Model Training Script
========================================================
Entraîne un modèle de classification pour détecter le spam dans les SMS/emails.

Author: KHADY AYMANE
Module: Intelligence Artificielle & Machine Learning
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins des fichiers
DATASET_PATH = "spam_dataset.csv"
MODEL_OUTPUT = "spam_detector_model.pkl"
VECTORIZER_OUTPUT = "tfidf_vectorizer.pkl"
REPORT_OUTPUT = "training_report.json"

# Paramètres du modèle
TEST_SIZE = 0.2
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def print_section(title):
    """Affiche un titre de section formaté."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_dataset(path):
    """Charge et affiche les informations du dataset."""
    print_section("📦 CHARGEMENT DU DATASET")
    
    try:
        # Essayer de charger le CSV
        df = pd.read_csv(path)
        print(f"✅ Dataset chargé avec succès")
        print(f"   Fichier: {path}")
        print(f"   Taille: {df.shape[0]} messages, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        print(f"⚠️  Dataset non trouvé. Création d'un dataset de démonstration...")
        return create_sample_dataset()


def create_sample_dataset():
    """
    Crée un dataset de démonstration si le fichier n'existe pas.
    Contient des exemples réalistes de spam et ham en anglais et français.
    """
    data = {
        'message': [
            # Spam - Offres frauduleuses
            "Congratulations! You won a FREE iPhone 6. Click here now: http://click.me",
            "You are a lucky winner! Claim your prize of £1000 NOW",
            "BUY CHEAP MEDICATIONS NOW - No prescription needed! Fast delivery",
            "URGENT: Verify your PayPal account NOW or it will be closed",
            "Click here to claim your FREE Samsung Galaxy S20! Limited offer",
            "You have been selected to receive $5000. Reply NOW with your bank details",
            "HOT GIRL NEAR YOU! Click here to chat: http://xxx.com",
            "Get rich quick! Earn $5000/week working from home",
            "Your account will be suspended unless you verify now: http://fake.bank",
            "Weight Loss Pill BREAKTHROUGH! Lose 50lbs in 2 weeks. Order today!",
            "FREE MONEY TRANSFER! Send us $100 and receive $1000 back",
            "Click here to download free movies and games - NO virus guaranteed",
            "Your tax refund is ready! Claim it here: http://irs.fake",
            "Viagra, Cialis, Levitra - buy online NO prescription needed",
            "WORK FROM HOME - No experience needed! $5000/month guaranteed",
            
            # Spam - Messages publicitaires aggressifs
            "LIMITED TIME OFFER! 70% OFF everything. Buy NOW http://sale.com",
            "You have 1 hour to claim this offer before it expires!",
            "Click here now to participate in our exclusive survey and win!",
            "Our customer service is struggling. Take survey, get $50!",
            "Last chance to take advantage of these amazing deals!",
            
            # Ham - Messages légitimes
            "Hey, how are you doing today?",
            "Thanks for the update. See you tomorrow at the meeting.",
            "I'll call you later this evening to discuss the project.",
            "Meeting rescheduled to 2pm tomorrow. Please confirm.",
            "Your appointment is confirmed for Dec 15 at 3pm.",
            "Just checking in - how's everything going?",
            "Can you send me the report by end of day?",
            "Great work on the presentation! Really impressed.",
            "Are you free for coffee this weekend?",
            "Update: We're implementing the new system next week.",
            "Your package has been delivered. Signature required.",
            "Thanks for your help with this. Really appreciated!",
            "I'm running 15 minutes late. See you soon.",
            "The files are ready. Download them from the shared folder.",
            "Let's catch up soon. It's been too long!",
            
            # Ham - Messages courts et normaux
            "Ok, see you at 5.",
            "Thanks!",
            "Sounds good to me.",
            "I agree, let's do it.",
            "Call me when you get a chance.",
            "What time works for you?",
            "Looking forward to it!",
            "Got it, thanks for letting me know.",
            "Will do!",
            "See you there!",
            "Perfect, thanks.",
            "On my way.",
            "Confirmed.",
            "Understood.",
            "All good here.",
        ],
        'label': [
            # Spam (15)
            'spam', 'spam', 'spam', 'spam', 'spam',
            'spam', 'spam', 'spam', 'spam', 'spam',
            'spam', 'spam', 'spam', 'spam', 'spam',
            
            # Spam (5)
            'spam', 'spam', 'spam', 'spam', 'spam',
            
            # Ham (15)
            'ham', 'ham', 'ham', 'ham', 'ham',
            'ham', 'ham', 'ham', 'ham', 'ham',
            'ham', 'ham', 'ham', 'ham', 'ham',
            
            # Ham (10)
            'ham', 'ham', 'ham', 'ham', 'ham',
            'ham', 'ham', 'ham', 'ham', 'ham',
            'ham', 'ham', 'ham', 'ham', 'ham',
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset de démonstration créé: {DATASET_PATH}")
    return df


def analyze_dataset(df):
    """Analyse et affiche les statistiques du dataset."""
    print_section("📊 ANALYSE EXPLORATOIRE")
    
    print("\n📋 Informations générales:")
    print(f"   Total de messages: {len(df)}")
    print(f"   Colonnes: {list(df.columns)}")
    
    print("\n📈 Distribution des classes:")
    class_dist = df['label'].value_counts()
    for label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {label.upper()}: {count} messages ({percentage:.1f}%)")
    
    print("\n📏 Statistiques textuelles:")
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    
    print(f"   Longueur moyenne: {df['message_length'].mean():.0f} caractères")
    print(f"   Longueur max: {df['message_length'].max()} caractères")
    print(f"   Mots moyens: {df['word_count'].mean():.1f} mots")
    
    return df


def prepare_data(df):
    """
    Prépare les données pour l'entraînement.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    print_section("🔧 PRÉPARATION DES DONNÉES")
    
    # Séparation features/target
    X = df['message']
    y = df['label']
    
    print(f"✅ Messages: {len(X)}")
    print(f"✅ Classes: {y.unique()}")
    
    # Conversion des labels en binaire (0=ham, 1=spam)
    y_binary = (y == 'spam').astype(int)
    print(f"✅ Labels convertis en binaire (ham=0, spam=1)")
    
    # Vectorisation TF-IDF
    print(f"\n🔤 Vectorisation TF-IDF:")
    print(f"   Max features: {TFIDF_MAX_FEATURES}")
    print(f"   N-grams: {TFIDF_NGRAM_RANGE}")
    
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95
    )
    
    X_tfidf = vectorizer.fit_transform(X)
    print(f"✅ Features extraites: {X_tfidf.shape[1]} features")
    
    # Train/Test split
    print(f"\n📂 Séparation Train/Test:")
    print(f"   Train: {100-int(TEST_SIZE*100)}%")
    print(f"   Test: {int(TEST_SIZE*100)}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_binary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary
    )
    
    print(f"✅ Train set: {X_train.shape[0]} messages")
    print(f"✅ Test set: {X_test.shape[0]} messages")
    
    return X_train, X_test, y_train, y_test, vectorizer


def train_model(X_train, y_train):
    """Entraîne le modèle Logistic Regression."""
    print_section("🤖 ENTRAÎNEMENT DU MODÈLE")
    
    print("Algorithme: Logistic Regression")
    print("   Raison du choix:")
    print("   - Entraînement rapide")
    print("   - Interprétabilité haute")
    print("   - Performance excellente pour la classification binaire")
    print("   - Probabilités bien calibrées")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        C=1.0,
        class_weight='balanced'
    )
    
    print("\n🎯 Entraînement en cours...")
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Évalue le modèle et retourne les métriques."""
    print_section("📊 ÉVALUATION DU MODÈLE")
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilités
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques Train
    print("\n🎓 PERFORMANCE SUR LE TRAINING SET:")
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    print(f"   Accuracy:  {train_accuracy*100:.2f}%")
    print(f"   Precision: {train_precision*100:.2f}%")
    print(f"   Recall:    {train_recall*100:.2f}%")
    print(f"   F1-Score:  {train_f1*100:.2f}%")
    
    # Métriques Test
    print("\n🔬 PERFORMANCE SUR LE TEST SET (Important!):")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"   Accuracy:  {test_accuracy*100:.2f}% ✅")
    print(f"   Precision: {test_precision*100:.2f}% ✅")
    print(f"   Recall:    {test_recall*100:.2f}% ✅")
    print(f"   F1-Score:  {test_f1*100:.2f}% ✅")
    print(f"   AUC-ROC:   {test_auc:.4f} ✅")
    
    # Matrice de confusion
    print("\n📋 MATRICE DE CONFUSION:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"   VP (True Positives):  {cm[1,1]}")
    print(f"   FN (False Negatives): {cm[1,0]}")
    print(f"   FP (False Positives): {cm[0,1]}")
    print(f"   VN (True Negatives):  {cm[0,0]}")
    
    # Rapport détaillé
    print("\n📊 RAPPORT DE CLASSIFICATION:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=['Ham (Légitime)', 'Spam'],
        digits=4
    ))
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'train_precision': float(train_precision),
        'train_recall': float(train_recall),
        'train_f1': float(train_f1),
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'confusion_matrix': cm.tolist(),
        'model_type': 'Logistic Regression',
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
    }
    
    return metrics


def save_model(model, vectorizer, metrics):
    """Sauvegarde le modèle et la vectorizer."""
    print_section("💾 SAUVEGARDE DES ARTEFACTS")
    
    # Sauvegarde du modèle
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Modèle sauvegardé: {MODEL_OUTPUT}")
    
    # Sauvegarde de la vectorizer
    with open(VECTORIZER_OUTPUT, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer sauvegardé: {VECTORIZER_OUTPUT}")
    
    # Sauvegarde du rapport
    with open(REPORT_OUTPUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Rapport sauvegardé: {REPORT_OUTPUT}")


def test_predictions(model, vectorizer):
    """Teste le modèle avec des exemples."""
    print_section("🧪 TEST AVEC DES EXEMPLES")
    
    test_messages = [
        "Hey, how are you doing today?",
        "Click here to win FREE MONEY now!!!",
        "Meeting tomorrow at 2pm. Confirmed?",
        "You won $1000! Claim your prize here: http://xxx.com",
        "Thanks for your help!",
    ]
    
    print("Prédictions:")
    for msg in test_messages:
        # Vectoriser le message
        msg_vec = vectorizer.transform([msg])
        
        # Prédire
        prediction = model.predict(msg_vec)[0]
        proba = model.predict_proba(msg_vec)[0]
        
        label = "🔴 SPAM" if prediction == 1 else "🟢 HAM"
        spam_prob = proba[1] * 100
        
        print(f"\n   {label} ({spam_prob:.1f}% spam)")
        print(f"   Message: \"{msg[:60]}...\"")


def main():
    """Fonction principale du script d'entraînement."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🛡️  SPAM DETECTOR - TRAINING SCRIPT  🛡️" + " " * 15 + "║")
    print("║" + " " * 68 + "║")
    print("║" + " " * 12 + "Détection de Spam avec Machine Learning" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    # 1. Charger le dataset
    df = load_dataset(DATASET_PATH)
    
    # 2. Analyser les données
    df = analyze_dataset(df)
    
    # 3. Préparer les données
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    
    # 4. Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # 5. Évaluer le modèle
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # 6. Sauvegarder les artefacts
    save_model(model, vectorizer, metrics)
    
    # 7. Tester avec des exemples
    test_predictions(model, vectorizer)
    
    # Message final
    print_section("✅ ENTRAÎNEMENT TERMINÉ")
    print("\nLes fichiers suivants ont été générés:")
    print(f"  1. {MODEL_OUTPUT} - Modèle entraîné")
    print(f"  2. {VECTORIZER_OUTPUT} - Vectorizer TF-IDF")
    print(f"  3. {REPORT_OUTPUT} - Rapport de performance")
    print("\nProchaines étapes:")
    print("  1. Lancer l'API: python app.py")
    print("  2. Ouvrir l'interface web: open index.html")
    print("\n")


if __name__ == "__main__":
    main()
