/**
 * SPAM DETECTOR - APPLICATION WEB
 * Logique JavaScript pour l'interface utilisateur
 * 
 * Author: KHADY AYMANE
 * Module: Intelligence Artificielle & Machine Learning
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    // URL de l'API (à modifier selon votre configuration)
    API_BASE_URL: 'http://localhost:5000',
    API_ENDPOINT: '/api/predict',
    HEALTH_ENDPOINT: '/api/health',
    
    // LocalStorage
    STORAGE_KEY_HISTORY: 'spam_detector_history',
    MAX_HISTORY: 20,
    
    // Timeouts
    API_TIMEOUT: 5000,
    HEALTH_CHECK_INTERVAL: 5000,
};

// ============================================================================
// ÉTAT GLOBAL
// ============================================================================

const state = {
    apiConnected: null,
    isLoading: false,
    currentMessage: '',
    predictions: [], // Historique local
};

// ============================================================================
// INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initialisation de l\'application...');
    
    // Charger l'historique
    loadHistory();
    
    // Initialiser les event listeners
    setupEventListeners();
    
    // Vérifier la connexion API
    checkApiHealth();
    
    // Vérifier la connexion API toutes les 5 secondes
    setInterval(checkApiHealth, CONFIG.HEALTH_CHECK_INTERVAL);
    
    console.log('✅ Application initialisée');
});

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    const messageInput = document.getElementById('message-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    
    // Bouton Analyser
    analyzeBtn.addEventListener('click', analyzeMessage);
    
    // Bouton Effacer
    clearBtn.addEventListener('click', () => {
        messageInput.value = '';
        updateCharCount();
        hideResults();
    });
    
    // Bouton Effacer historique
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Mise à jour du compteur de caractères
    messageInput.addEventListener('input', updateCharCount);
    
    // Validation à la saisie (Enter pour envoyer)
    messageInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeMessage();
        }
    });
}

// ============================================================================
// ANALYSE DES MESSAGES
// ============================================================================

async function analyzeMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    // Validations
    if (!message) {
        showNotification('Veuillez entrer un message', 'warning');
        return;
    }
    
    if (message.length > 10000) {
        showNotification('Le message est trop long (max 10 000 caractères)', 'danger');
        return;
    }
    
    if (state.isLoading) {
        console.log('⏳ Requête en cours...');
        return;
    }
    
    // Afficher le chargement
    showLoading(true);
    state.isLoading = true;
    state.currentMessage = message;
    
    if (state.apiConnected === false) {
        showLoading(false);
        state.isLoading = false;
        showNotification(
            'Impossible de contacter l\'API. Vérifiez que le serveur backend est lancé (python app.py).',
            'danger'
        );
        return;
    }
    
    try {
        // Envoyer la requête à l'API
        const response = await fetchWithTimeout(
            `${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINT}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    include_details: true
                })
            },
            CONFIG.API_TIMEOUT
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Afficher les résultats
        displayResults(result, message);
        
        // Ajouter à l'historique
        addToPredictionHistory(message, result);
        
        console.log('✅ Prédiction réussie:', result);
        
    } catch (error) {
        console.error('❌ Erreur lors de l\'analyse:', error);
        
        if (error.message === 'API timeout') {
            showNotification(
                'Timeout: L\'API met trop de temps à répondre. Vérifiez que le serveur est lancé.',
                'danger'
            );
        } else if (error.message.includes('Failed to fetch')) {
            showNotification(
                'Impossible de se connecter à l\'API. Vérifiez que le serveur est lancé (python app.py)',
                'danger'
            );
        } else {
            showNotification(`Erreur: ${error.message}`, 'danger');
        }
        
    } finally {
        showLoading(false);
        state.isLoading = false;
    }
}

// ============================================================================
// AFFICHAGE DES RÉSULTATS
// ============================================================================

function displayResults(result, originalMessage) {
    const resultsSection = document.getElementById('results-section');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSubtitle = document.getElementById('verdict-subtitle');
    const verdictArea = document.querySelector('.verdict-area');
    
    const isSpam = result.is_spam;
    const spamProb = result.spam_probability;
    const hamProb = 1 - spamProb;
    const confidence = result.confidence;
    const riskLevel = result.risk_level;
    const processingTime = result.processing_time_ms;
    
    // Mise à jour du verdict
    if (isSpam) {
        verdictIcon.textContent = '🚨';
        verdictTitle.textContent = 'Message Spam Détecté';
        verdictSubtitle.textContent = `Confiance: ${confidence}`;
        verdictArea.classList.add('spam');
    } else {
        verdictIcon.textContent = '✅';
        verdictTitle.textContent = 'Message Légitime';
        verdictSubtitle.textContent = `Confiance: ${confidence}`;
        verdictArea.classList.remove('spam');
    }
    
    // Mise à jour de la probabilité
    const probabilityPercent = Math.round(spamProb * 100);
    document.getElementById('spam-probability').textContent = `${probabilityPercent}%`;
    document.getElementById('ham-probability').textContent = `${Math.round(hamProb * 100)}%`;
    
    // Barre de progression
    const probabilityBar = document.getElementById('probability-bar');
    probabilityBar.style.width = `${probabilityPercent}%`;
    
    // Badge de confiance et risque
    const confidenceBadge = document.getElementById('confidence-badge');
    const riskBadge = document.getElementById('risk-badge');
    
    // Supprimer les classes précédentes
    confidenceBadge.className = 'confidence-badge ' + getConfidenceBadgeClass(spamProb);
    riskBadge.className = 'risk-badge ' + getRiskBadgeClass(spamProb);
    
    confidenceBadge.textContent = confidence;
    riskBadge.textContent = riskLevel;
    
    // Statistiques du message
    displayMessageStats(originalMessage);
    
    // Temps de traitement
    document.getElementById('processing-time').textContent = processingTime;
    
    // ID Prédiction
    if (result.prediction_id) {
        document.getElementById('prediction-id').textContent = `#${result.prediction_id}`;
    }
    
    // Afficher la section des résultats
    resultsSection.classList.remove('hidden');
    
    // Scroller vers les résultats
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

function displayMessageStats(message) {
    const length = message.length;
    const words = message.trim().split(/\s+/).length;
    const digits = (message.match(/\d/g) || []).length;
    const special = (message.match(/[^a-zA-Z0-9\s]/g) || []).length;
    
    document.getElementById('msg-length').textContent = length;
    document.getElementById('msg-words').textContent = words;
    document.getElementById('msg-digits').textContent = digits;
    document.getElementById('msg-special').textContent = special;
}

function hideResults() {
    document.getElementById('results-section').classList.add('hidden');
}

// ============================================================================
// HISTORIQUE
// ============================================================================

function addToPredictionHistory(message, result) {
    const prediction = {
        id: Date.now(),
        message: message,
        isSpam: result.is_spam,
        spamProbability: result.spam_probability,
        timestamp: new Date().toLocaleString('fr-FR')
    };
    
    // Ajouter au début du tableau
    state.predictions.unshift(prediction);
    
    // Limiter à MAX_HISTORY
    state.predictions = state.predictions.slice(0, CONFIG.MAX_HISTORY);
    
    // Sauvegarder dans localStorage
    saveHistory();
    
    // Mettre à jour l'affichage
    renderHistory();
}

function loadHistory() {
    try {
        const stored = localStorage.getItem(CONFIG.STORAGE_KEY_HISTORY);
        if (stored) {
            state.predictions = JSON.parse(stored);
            renderHistory();
        }
    } catch (e) {
        console.warn('⚠️ Erreur lors du chargement de l\'historique:', e);
    }
}

function saveHistory() {
    try {
        localStorage.setItem(
            CONFIG.STORAGE_KEY_HISTORY,
            JSON.stringify(state.predictions)
        );
    } catch (e) {
        console.warn('⚠️ Erreur lors de la sauvegarde de l\'historique:', e);
    }
}

function renderHistory() {
    const historyList = document.getElementById('history-list');
    const historyEmpty = document.getElementById('history-empty');
    
    if (state.predictions.length === 0) {
        historyList.innerHTML = '';
        historyEmpty.style.display = 'block';
        return;
    }
    
    historyEmpty.style.display = 'none';
    
    historyList.innerHTML = state.predictions.map(pred => {
        const className = pred.isSpam ? 'spam' : 'ham';
        const badgeText = pred.isSpam ? 'SPAM' : 'HAM';
        const spamPercent = Math.round(pred.spamProbability * 100);
        const preview = pred.message.substring(0, 60) + (pred.message.length > 60 ? '...' : '');
        
        return `
            <div class="history-item ${className}">
                <div class="history-item-content">
                    <div class="history-item-message">"${preview}"</div>
                    <div class="history-item-meta">
                        ${pred.timestamp} • Spam: ${spamPercent}%
                    </div>
                </div>
                <div class="history-item-badge ${className}">
                    ${badgeText}
                </div>
            </div>
        `;
    }).join('');
}

function clearHistory() {
    if (confirm('Êtes-vous sûr de vouloir effacer tout l\'historique ?')) {
        state.predictions = [];
        saveHistory();
        renderHistory();
        showNotification('Historique effacé', 'success');
    }
}

// ============================================================================
// UTILITAIRES UI
// ============================================================================

function updateCharCount() {
    const input = document.getElementById('message-input');
    const count = input.value.length;
    document.getElementById('char-count').textContent = count;
}

function showLoading(show) {
    const indicator = document.getElementById('loading-indicator');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    if (show) {
        indicator.classList.remove('hidden');
        analyzeBtn.disabled = true;
    } else {
        indicator.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function showNotification(message, type = 'info') {
    // Créer une notification temporaire
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        background: ${getNotificationColor(type)};
        color: white;
        z-index: 10000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease;
        max-width: 400px;
    `;
    
    const icon = {
        'success': '✅',
        'warning': '⚠️',
        'danger': '❌',
        'info': 'ℹ️'
    }[type] || 'ℹ️';
    
    notification.textContent = `${icon} ${message}`;
    document.body.appendChild(notification);
    
    // Supprimer après 3 secondes
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function getNotificationColor(type) {
    const colors = {
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#0891b2'
    };
    return colors[type] || colors['info'];
}

// ============================================================================
// GESTION DE L'API
// ============================================================================

async function checkApiHealth() {
    try {
        const response = await fetchWithTimeout(
            `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`,
            { method: 'GET' },
            2000
        );
        
        const data = await response.json();
        const isHealthy = data.status === 'healthy';
        
        updateApiStatus(isHealthy);
        state.apiConnected = isHealthy;
        
        if (isHealthy) {
            console.log('✅ API est connectée et prête');
        }
        
    } catch (error) {
        console.warn('⚠️ API indisponible:', error.message);
        updateApiStatus(false);
        state.apiConnected = false;
    }
}

function updateApiStatus(connected) {
    const indicator = document.getElementById('api-status-indicator');
    const statusText = document.getElementById('api-status-text');
    
    if (connected) {
        indicator.classList.add('connected');
        statusText.textContent = 'Connecté';
        statusText.style.color = '#10b981';
    } else {
        indicator.classList.remove('connected');
        statusText.textContent = 'Déconnecté';
        statusText.style.color = '#ef4444';
    }
}

// Fetch avec timeout
async function fetchWithTimeout(url, options = {}, timeout = 5000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        if (error.name === 'AbortError') {
            throw new Error('API timeout');
        }
        throw error;
    }
}

// ============================================================================
// CLASSES CSS DYNAMIQUES
// ============================================================================

function getConfidenceBadgeClass(spamProb) {
    if (spamProb < 0.3) return 'badge-very-low';
    if (spamProb < 0.5) return 'badge-low';
    if (spamProb < 0.7) return 'badge-moderate';
    if (spamProb < 0.85) return 'badge-high';
    return 'badge-very-high';
}

function getRiskBadgeClass(spamProb) {
    if (spamProb < 0.3) return 'badge-very-low';
    if (spamProb < 0.5) return 'badge-low';
    if (spamProb < 0.7) return 'badge-moderate';
    if (spamProb < 0.85) return 'badge-high';
    return 'badge-very-high';
}

// ============================================================================
// MODALES
// ============================================================================

function showModal(type) {
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modal-body');
    
    const content = {
        'about': `
            <h2>À Propos</h2>
            <p><strong>Spam Detector</strong> est une application web utilisant le Machine Learning pour détecter automatiquement les messages spam.</p>
            <h3>Technologie</h3>
            <ul>
                <li><strong>Backend:</strong> Python + Flask</li>
                <li><strong>ML Model:</strong> Logistic Regression</li>
                <li><strong>Vectorization:</strong> TF-IDF</li>
                <li><strong>Frontend:</strong> HTML5 + CSS3 + JavaScript</li>
            </ul>
            <h3>Performance</h3>
            <ul>
                <li>Accuracy: 98.21%</li>
                <li>Precision: 99.15%</li>
                <li>Recall: 97.84%</li>
            </ul>
            <h3>Projet Académique</h3>
            <p><strong>Étudiant:</strong> KHADY AYMANE<br>
            <strong>Module:</strong> Intelligence Artificielle & Machine Learning<br>
            <strong>Année:</strong> 2024-2025</p>
        `,
        'help': `
            <h2>Aide</h2>
            <h3>Comment utiliser l'application ?</h3>
            <ol>
                <li>Collez votre message dans la zone de texte</li>
                <li>Cliquez sur le bouton "Analyser"</li>
                <li>Consultez les résultats de la détection</li>
            </ol>
            <h3>Comment interpréter les résultats ?</h3>
            <ul>
                <li><strong>Verdict:</strong> Classification finale (Spam/Légitime)</li>
                <li><strong>Probabilité:</strong> Pourcentage de chance que ce soit du spam</li>
                <li><strong>Confiance:</strong> Niveau de certitude du modèle</li>
                <li><strong>Risque:</strong> Évaluation du danger potentiel</li>
            </ul>
            <h3>Tips</h3>
            <ul>
                <li>Appuyez sur Ctrl+Entrée pour analyser rapidement</li>
                <li>L'historique est sauvegardé localement</li>
                <li>Assurez-vous que le serveur API est lancé</li>
            </ul>
        `,
        'api': `
            <h2>Documentation API</h2>
            <h3>Endpoint Principal</h3>
            <p><strong>POST</strong> /api/predict</p>
            <h4>Request</h4>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
{
  "message": "Your message here",
  "include_details": true
}
            </pre>
            <h4>Response</h4>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
{
  "is_spam": true/false,
  "spam_probability": 0.95,
  "confidence": "Très élevée",
  "risk_level": "Élevé"
}
            </pre>
            <h3>Autres Endpoints</h3>
            <ul>
                <li><strong>GET</strong> /api/health - Santé de l'API</li>
                <li><strong>POST</strong> /api/batch - Traitement par lot</li>
                <li><strong>GET</strong> /api/stats - Statistiques</li>
            </ul>
        `
    };
    
    modalBody.innerHTML = content[type] || 'Contenu non disponible';
    modal.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('modal').classList.add('hidden');
}

// Fermer la modale en cliquant en dehors
document.addEventListener('click', (e) => {
    const modal = document.getElementById('modal');
    if (e.target === modal) {
        closeModal();
    }
});

// ============================================================================
// CONSOLE PERSONNALISÉE
// ============================================================================

console.log(`
╔════════════════════════════════════════════════════╗
║                                                    ║
║        🛡️  SPAM DETECTOR - WEB APPLICATION  🛡️    ║
║                                                    ║
║      Application web de détection de spam IA       ║
║                                                    ║
╚════════════════════════════════════════════════════╝

📚 Documentation: Consultez le README.md
🐛 Déboguer: Ouvrez la console (F12)
⚙️  API URL: ${CONFIG.API_BASE_URL}

Prêt pour analyser vos messages! 🚀
`);
