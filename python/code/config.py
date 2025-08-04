import os

# --- Model Configuration ---
TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64 # Reverted to 128 for speed
DISTILBERT_HIDDEN_SIZE = 768 # DistilBERT's hidden size remains 768

# --- Training Parameters ---
INITIAL_LEARNING_RATE = 2e-5
INITIAL_EPOCHS = 5
BATCH_SIZE = 32

# --- Adversarial Training Parameters ---
NUM_ADVERSARIAL_ITERATIONS = 5
NUM_ATTACKS_PER_ITERATION = 50 # Number of adversarial examples to generate per iteration
ADVERSARIAL_LEARNING_RATE = 1e-5 # Slightly lower LR for fine-tuning during adversarial training
EPOCHS_PER_ADVERSARIAL_ITERATION = 2 # Fewer epochs per fine-tuning step

# --- Data Features ---
TEXT_FEATURE = 'sms_content_cleaned_for_nlp'
TARGET_FEATURE = 'is_phishing'

NUMERICAL_FEATURES_FOR_SCALING = [
    'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
    'phishing_keywords_count', 'domain_age_days',
    'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
    'word_count', 'avg_word_length',
    'digits_to_length_ratio', 'special_chars_to_length_ratio', 'phishing_keywords_to_word_count_ratio'
]
BOOLEAN_FEATURES_AS_NUMERICAL = [
    'has_url', 'has_email', 'has_phone_number', 'is_ip_address_url',
    'url_and_ip', 'url_and_suspicious_tld', 'urgent_and_url'
]
CATEGORICAL_FEATURES_FOR_OHE = [
    'domain_encoded', 'tld_encoded', 'subdomain_encoded', 'url_subcategory_encoded',
    'domain_registrar_encoded', 'sender_encoded', 'sender_type_encoded',
    'brand_encoded', 'message_category_encoded', 'dataset_source_encoded'
]

# --- Directory Paths (relative to the project root where main.py is) ---
DATA_DIR = os.path.join('..', 'data', 'generated')
MODEL_DIR = os.path.join('..', 'model')
EXPORTED_MODEL_DIR = os.path.join(MODEL_DIR, 'exported') # For Android assets

# --- Checkpoint File Paths ---
# Data preparation checkpoint
UNIFIED_DATA_PATH = os.path.join(DATA_DIR, 'unified_phishing_sms_dataset_processed.csv')

# Model checkpoints
INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, 'initial_hybrid_model.keras')
ROBUST_MODEL_PREFIX = os.path.join(MODEL_DIR, 'robust_hybrid_model_iteration_') # Suffix with iteration number
FINAL_ROBUST_MODEL_PATH = os.path.join(MODEL_DIR, 'final_robust_hybrid_model.keras')
FINAL_TFLITE_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, 'final_robust_hybrid_model.tflite')

# Adversarial examples checkpoints
ADVERSARIAL_EXAMPLES_DIR = os.path.join(DATA_DIR, 'adversarial_examples')
ADVERSARIAL_EXAMPLES_PREFIX = os.path.join(ADVERSARIAL_EXAMPLES_DIR, 'adv_examples_iter_') # Suffix with iteration number

# Training history/metrics
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')

# Plot saving paths
PLOTS_DIR = os.path.join(MODEL_DIR, 'plots')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
ROC_CURVE_PLOT_PATH = os.path.join(PLOTS_DIR, 'roc_curve.png')
TRAINING_ACC_LOSS_PLOT_PATH = os.path.join(PLOTS_DIR, 'training_acc_loss.png')
ADVERSARIAL_ACC_HISTORY_PLOT_PATH = os.path.join(PLOTS_DIR, 'adversarial_acc_history.png')

# Exported Android assets paths
ANDROID_VOCAB_PATH = os.path.join(EXPORTED_MODEL_DIR, 'vocab.txt')
ANDROID_SCALER_PARAMS_PATH = os.path.join(EXPORTED_MODEL_DIR, 'scaler_params.json')
ANDROID_ENCODER_PARAMS_PATH = os.path.join(EXPORTED_MODEL_DIR, 'encoder_params.json')
ANDROID_FEATURE_ORDER_PATH = os.path.join(EXPORTED_MODEL_DIR, 'structured_feature_order.json')


# --- Configuration for Data Preprocessing ---
NAME_DATASET1 = 'spam.csv'
NAME_DATASET2 = 'Dataset_5971.csv'
NAME_DATASET3 = 'analysisdataset.csv'


PHISHING_KEYWORDS = [
    'account', 'verify', 'update', 'security', 'password', 'login', 'click',
    'link', 'urgent', 'alert', 'suspicious', 'bank', 'credit', 'card',
    'confirm', 'fraud', 'prize', 'winner', 'claim', 'deliver', 'tracking',
    'invoice', 'payment', 'transfer', 'dear customer', 'congratulations',
    'restricted', 'action', 'required', 'immediately', 'now', 'blocked',
    'suspended', 'compromised', 'unusual', 'activity', 'verify', 'confirm',
    'secure', 'attention', 'important', 'warning', 'error', 'problem',
    'delivery', 'parcel', 'shipment', 'order', 'transaction', 'invoice',
    'refund', 'tax', 'irs', 'hmrc', 'government', 'fine', 'penalty',
    'lottery', 'winnings', 'gift', 'coupon', 'free', 'offer', 'exclusive',
    'limited time', 'expires', 'congratulations', 'selected', 'eligible',
    'prize', 'reward', 'cash', 'money', 'payment', 'transfer', 'deposit',
    'withdraw', 'loan', 'credit', 'debit', 'card', 'pin', 'atm', 'balance',
    'statement', 'bill', 'due', 'overdue', 'invoice', 'receipt', 'charge',
    'transaction', 'purchase', 'order', 'shipping', 'tracking', 'delivery',
    'package', 'shipment', 'dispatch', 'courier', 'post', 'mail',
    'support', 'customer service', 'help desk', 'technical support',
    'service', 'issue', 'problem', 'fix', 'resolve', 'restore', 'recover',
    'reset', 'reactivate', 'unlock', 'disable', 'enable', 'access',
    'personal', 'information', 'details', 'data', 'credentials', 'identity',
    'ssn', 'dob', 'address', 'phone', 'email', 'username', 'password',
    'otp', 'code', 'token', 'verification', 'authentication', 'authorization',
    'security', 'fraud', 'scam', 'spam', 'malicious', 'suspicious', 'unsafe',
    'warning', 'alert', 'notice', 'notification', 'message', 'text', 'sms',
    'call', 'dial', 'visit', 'link', 'website', 'url', 'portal', 'page',
    'form', 'survey', 'update', 'upgrade', 'install', 'download', 'app',
    'software', 'program', 'virus', 'malware', 'trojan', 'ransomware',
    'exploit', 'vulnerability', 'breach', 'leak', 'data breach',
    'urgent action required', 'account suspended', 'click here', 'verify your account',
    'unusual login activity', 'security alert', 'prize winner', 'claim your reward',
    'delivery failed', 'track your package', 'payment pending', 'invoice attached',
    'tax refund', 'government grant', 'password reset', 'confirm your identity',
    'your account has been locked', 'update your details', 'suspicious transaction',
    'call us now', 'visit our website', 'download the app', 'important notice',
    'final warning', 'immediate action', 'security warning', 'phishing detected',
    'spam alert', 'malware detected', 'fraudulent activity', 'unauthorized access',
    'click the link', 'login to your account', 'bank alert', 'credit card fraud',
    'urgent message', 'delivery notification', 'package tracking', 'payment confirmation',
    'winning notification', 'gift card', 'free gift', 'limited offer',
    'customer support', 'technical issue', 'account recovery', 'password change',
    'security update', 'data verification', 'identity theft', 'social security number',
    'date of birth', 'personal data', 'login credentials', 'one-time password',
    'verification code', 'authentication required', 'authorization required',
    'security threat', 'fraud alert', 'scam alert', 'spam warning', 'malicious software',
    "suspicious activity", "unsafe link", "warning message", "alert notification",
    "text message", "call now", "dial number", "visit link", "website access",
    "portal login", "page access", "form submission", "survey link", "update software",
    "upgrade system", "install app", "download program", "virus alert", "malware detected",
    "trojan warning", "ransomware threat", "exploit vulnerability", "data breach detected"
]
