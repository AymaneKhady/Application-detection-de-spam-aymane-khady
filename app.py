#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPAM DETECTOR - REST API
=======================
API Flask pour la détection de spam en temps réel.

Endpoints:
  POST /api/predict - Prédire si un message est spam
  GET  /api/health  - Vérifier la santé de l'API
  GET  /          - Page d'accueil

Author: KHADY AYMANE
Module: Intelligence Artificielle & Machine Learning
Date: 2025
"""

from flask import Flask, request, jsonify, send_file
import pickle
import os
import json
from datetime import datetime
import logging
from functools import wraps

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialiser Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration CORS simple
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Chemins des fichiers
MODEL_PATH = "spam_detector_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
LOG_FILE = "predictions.log"

# Variables globales
model = None
vectorizer = None
predictions_count = 0

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def log_prediction(message, is_spam, confidence):
    """Enregistre une prédiction dans le fichier log."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'message_length': len(message),
        'is_spam': bool(is_spam),
        'confidence': float(confidence)
    }
    
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement: {e}")


def get_confidence_level(spam_probability):
    """Retourne un niveau de confiance texte basé sur la probabilité."""
    if spam_probability < 0.3:
        return "Très faible"
    elif spam_probability < 0.5:
        return "Faible"
    elif spam_probability < 0.7:
        return "Modérée"
    elif spam_probability < 0.85:
        return "Élevée"
    else:
        return "Très élevée"


def get_risk_level(spam_probability):
    """Retourne un niveau de risque basé sur la probabilité."""
    if spam_probability < 0.3:
        return "Très faible"
    elif spam_probability < 0.5:
        return "Faible"
    elif spam_probability < 0.7:
        return "Modéré"
    elif spam_probability < 0.85:
        return "Élevé"
    else:
        return "Très élevé"


def require_api_key(f):
    """Décorateur pour vérifier la clé API (optionnel)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Pour ce projet, on désactive la vérification de clé API
        # En production, il faudrait vérifier: request.headers.get('X-API-Key')
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

def load_model():
    """Charge le modèle et la vectorizer au démarrage."""
    global model, vectorizer
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        logger.error("❌ Les fichiers du modèle n'existent pas!")
        logger.error(f"   Cherche: {MODEL_PATH} et {VECTORIZER_PATH}")
        logger.error("   Exécuter d'abord: python train_model.py")
        return False
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✅ Modèle chargé: {MODEL_PATH}")
        
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"✅ Vectorizer chargé: {VECTORIZER_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return False


# ============================================================================
# ROUTES API
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API."""
    return jsonify({
        'app': 'Spam Detector API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'POST /api/predict': 'Prédire si un message est spam',
            'GET /api/health': 'Vérifier la santé de l\'API',
            'GET /': 'Cette page'
        },
        'example': {
            'method': 'POST',
            'endpoint': '/api/predict',
            'payload': {
                'message': 'Click here to win FREE MONEY!!!',
                'include_details': True
            }
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de santé de l'API."""
    model_loaded = model is not None
    vectorizer_loaded = vectorizer is not None
    
    return jsonify({
        'status': 'healthy' if (model_loaded and vectorizer_loaded) else 'degraded',
        'model_loaded': model_loaded,
        'vectorizer_loaded': vectorizer_loaded,
        'timestamp': datetime.now().isoformat()
    }), 200 if (model_loaded and vectorizer_loaded) else 503


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Endpoint principal pour la prédiction de spam.
    
    POST /api/predict
    
    Request body:
    {
        "message": "Your message here",
        "include_details": true  (optionnel)
    }
    
    Response:
    {
        "message": "Your message here",
        "is_spam": true/false,
        "spam_probability": 0.95,
        "confidence": "Très élevée",
        "risk_level": "Très élevé",
        "processing_time_ms": 12
    }
    """
    global predictions_count
    
    # Vérifier que le modèle est chargé
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Le modèle n\'a pas pu être chargé. Exécutez: python train_model.py'
        }), 503
    
    # Gestion des CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    # Récupérer les données
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({
            'error': 'Missing required field: message',
            'example': {'message': 'Click here to win FREE MONEY!!!'}
        }), 400
    
    message = data.get('message', '').strip()
    include_details = data.get('include_details', False)
    
    # Valider le message
    if not message:
        return jsonify({
            'error': 'Message cannot be empty',
        }), 400
    
    if len(message) > 10000:
        return jsonify({
            'error': 'Message too long (max 10000 characters)',
        }), 400
    
    try:
        # Mesurer le temps
        import time
        start_time = time.time()
        
        # Vectoriser le message
        message_vec = vectorizer.transform([message])
        
        # Prédire
        prediction = model.predict(message_vec)[0]
        probabilities = model.predict_proba(message_vec)[0]
        
        spam_probability = float(probabilities[1])  # Probabilité de spam
        is_spam = bool(prediction == 1)
        
        processing_time = (time.time() - start_time) * 1000  # en ms
        
        # Incrémenter le compteur
        predictions_count += 1
        
        # Enregistrer la prédiction
        log_prediction(message, is_spam, spam_probability)
        
        # Préparer la réponse
        response = {
            'message': message[:100] + '...' if len(message) > 100 else message,
            'is_spam': is_spam,
            'spam_probability': round(spam_probability, 4),
            'confidence': get_confidence_level(spam_probability),
            'risk_level': get_risk_level(spam_probability),
            'processing_time_ms': round(processing_time, 2),
            'prediction_id': predictions_count
        }
        
        # Ajouter des détails si demandé
        if include_details:
            response['details'] = {
                'ham_probability': round(float(probabilities[0]), 4),
                'model_type': 'Logistic Regression',
                'features_count': message_vec.shape[1],
                'message_length': len(message),
                'word_count': len(message.split())
            }
        
        logger.info(f"Prédiction #{predictions_count}: {is_spam} (confiance: {spam_probability*100:.1f}%)")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """
    Endpoint pour traiter plusieurs messages à la fois.
    
    POST /api/batch
    
    Request body:
    {
        "messages": [
            "Message 1",
            "Message 2",
            "Message 3"
        ]
    }
    """
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    data = request.get_json()
    
    if not data or 'messages' not in data:
        return jsonify({
            'error': 'Missing required field: messages'
        }), 400
    
    messages = data.get('messages', [])
    
    if not isinstance(messages, list):
        return jsonify({
            'error': 'messages must be a list'
        }), 400
    
    if len(messages) > 100:
        return jsonify({
            'error': 'Maximum 100 messages per batch'
        }), 400
    
    try:
        results = []
        
        for msg in messages:
            if not isinstance(msg, str) or not msg.strip():
                continue
            
            msg = msg.strip()
            msg_vec = vectorizer.transform([msg])
            
            prediction = model.predict(msg_vec)[0]
            probabilities = model.predict_proba(msg_vec)[0]
            
            spam_probability = float(probabilities[1])
            
            results.append({
                'message': msg[:100],
                'is_spam': bool(prediction == 1),
                'spam_probability': round(spam_probability, 4)
            })
        
        return jsonify({
            'count': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Affiche les statistiques d'utilisation."""
    return jsonify({
        'predictions_total': predictions_count,
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# GESTION DES ERREURS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path,
        'method': request.method,
        'available_endpoints': ['/', '/api/health', '/api/predict', '/api/batch', '/api/stats']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur serveur: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🛡️  SPAM DETECTOR - REST API  🛡️")
    print("="*70 + "\n")
    
    # Charger le modèle
    print("📦 Chargement du modèle...")
    if not load_model():
        print("\n❌ Impossible de démarrer l'API sans le modèle!")
        print("   Exécutez d'abord: python train_model.py\n")
        exit(1)
    
    print("\n✅ Modèle chargé avec succès!")
    print("\n🚀 Démarrage de l'API...")
    print("   URL: http://localhost:5000")
    print("   Docs: Consultez README.md pour les endpoints\n")
    print("="*70 + "\n")
    
    # Lancer le serveur
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )
