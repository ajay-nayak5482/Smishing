import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from transformers import AutoTokenizer # Make sure AutoTokenizer is imported here
import numpy as np
from dataclasses import dataclass
import os
import re
from datetime import datetime
import tldextract
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Import configurations
from config import (
    TRANSFORMER_MODEL_NAME, MAX_LEN, TEXT_FEATURE, TARGET_FEATURE,
    NUMERICAL_FEATURES_FOR_SCALING, BOOLEAN_FEATURES_AS_NUMERICAL,
    CATEGORICAL_FEATURES_FOR_OHE, DATA_DIR, UNIFIED_DATA_PATH, PLOTS_DIR
)

# Initialize NLTK components globally within this module
try:
    _ = stopwords.words('english')
    _ = word_tokenize("test")
    _ = SentimentIntensityAnalyzer().polarity_scores("test")
except LookupError:
    import nltk
    print("Downloading NLTK data (stopwords, punkt, vader_lexicon)...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    print("NLTK data downloaded.")

STOP_WORDS = set(stopwords.words('english'))
VADER_ANALYZER = SentimentIntensityAnalyzer()

@dataclass
class ProcessedData:
    """
    A dataclass to hold all processed data tensors and preprocessor/tokenizer objects.
    Includes original X_train_val and X_test DataFrames for easier sampling later.
    """
    train_input_ids: tf.Tensor
    train_attention_mask: tf.Tensor
    X_train_structured_processed: tf.Tensor
    y_train: tf.Tensor

    val_input_ids: tf.Tensor
    val_attention_mask: tf.Tensor
    X_val_structured_processed: tf.Tensor
    y_val: tf.Tensor

    test_input_ids: tf.Tensor
    test_attention_mask: tf.Tensor
    X_test_structured_processed: tf.Tensor
    y_test: tf.Tensor
    
    tokenizer: AutoTokenizer
    preprocessor: ColumnTransformer
    
    # Store original DataFrames for sampling purposes in adversarial attacks
    X_train_original_df: pd.DataFrame
    X_test_original_df: pd.DataFrame

# --- Helper Functions for Feature Engineering (moved from data_preprocessing.py) ---

def extract_url_info(url_string):
    """Extracts domain, TLD, and subdomain using tldextract."""
    if pd.isna(url_string) or not isinstance(url_string, str) or not url_string.strip():
        return None, None, None, False
    
    if not re.match(r'^[a-zA-Z]+://', url_string):
        url_string = 'http://' + url_string

    try:
        extracted = tldextract.extract(url_string)
        is_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', extracted.domain))
        
        subdomain = extracted.subdomain if extracted.subdomain else None
        domain = extracted.domain if extracted.domain else None
        tld = extracted.suffix if extracted.suffix else None
        
        return domain, tld, subdomain, is_ip
    except Exception:
        return None, None, None, False

def calculate_domain_age(creation_date_str, last_update_date_str):
    """Calculates domain age in days from creation/update dates."""
    if pd.isna(creation_date_str) and pd.isna(last_update_date_str):
        return np.nan
    
    dates_to_consider = []
    if pd.notna(creation_date_str) and isinstance(creation_date_str, str):
        try:
            dates_to_consider.append(datetime.strptime(creation_date_str.split(' ')[0], '%Y-%m-%d'))
        except ValueError:
            pass
    if pd.notna(last_update_date_str) and isinstance(last_update_date_str, str):
        try:
            dates_to_consider.append(datetime.strptime(last_update_date_str.split(' ')[0], '%Y-%m-%d'))
        except ValueError:
            pass

    if not dates_to_consider:
        return np.nan
    
    earliest_date = min(dates_to_consider)
    return (datetime.now() - earliest_date).days

def clean_text_basic(text):
    """Basic text cleaning for NLP features (less aggressive for initial heuristics)."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_textual_heuristics(text, phishing_keywords):
    """Extracts various heuristic features from text."""
    if pd.isna(text):
        return 0, 0, 0, 0, 0
    
    text_cleaned_basic = clean_text_basic(text)

    sms_length = len(text_cleaned_basic)
    num_special_chars = sum(1 for char in text if char in string.punctuation)
    num_digits = sum(1 for char in text if char.isdigit())
    
    words = word_tokenize(text_cleaned_basic)
    num_all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    phishing_keywords_count = sum(1 for keyword in phishing_keywords if keyword in text_cleaned_basic)
    
    return sms_length, num_special_chars, num_digits, num_all_caps_words, phishing_keywords_count

def clean_and_normalize_text_for_nlp(text):
    """
    Performs comprehensive cleaning and normalization for NLP model input and advanced heuristic extraction.
    Removes stopwords.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = word_tokenize(text)
    words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(words)

def get_sentiment_scores(text):
    """Calculates VADER sentiment scores."""
    if pd.isna(text) or not text.strip():
        return 0.0, 0.0, 0.0, 0.0 # neg, neu, pos, compound
    scores = VADER_ANALYZER.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

# --- ADDED: tokenize_text function ---
def tokenize_text(texts, tokenizer, max_len):
    """
    Tokenizes a list of texts for transformer input.
    Returns input_ids, attention_mask.
    """
    texts = texts.astype(str).fillna('').tolist()
    encodings = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='tf'
    )
    return encodings['input_ids'], encodings['attention_mask']
# --- END ADDED ---


# --- Main Data Integration and Preprocessing Function ---

def integrate_and_preprocess_datasets(
    file_path_d1, file_path_d2, file_path_d3,
    phishing_keywords, output_path=UNIFIED_DATA_PATH
):
    """
    Loads, preprocesses, integrates all three datasets, and performs feature engineering.
    Saves the processed DataFrame to `output_path`.
    """
    if os.path.exists(output_path):
        print(f"Loading unified and processed data from checkpoint: {output_path}")
        unified_df = pd.read_csv(output_path)
        print(f"Loaded {len(unified_df)} rows from checkpoint.")
        return unified_df

    print("--- Starting Dataset Integration and Preprocessing ---")

    # Load Dataset 1 (assuming it's tab-separated based on previous code)
    try:
        # Assuming dataset1.csv has no header and is tab-separated as per earlier code
        df1 = pd.read_csv(file_path_d1, sep='\t', header=None, names=['v1', 'v2'], encoding='latin-1')
        df1['dataset_source'] = 'D1'
        print(f"Dataset 1 loaded: {len(df1)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 1 not found at {file_path_d1}. Please ensure it exists.")
        return None # Or create dummy data as fallback

    # Load Dataset 2
    try:
        df2 = pd.read_csv(file_path_d2)
        df2['dataset_source'] = 'D2'
        print(f"Dataset 2 loaded: {len(df2)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 2 not found at {file_path_d2}. Please ensure it exists.")
        return None

    # Load Dataset 3
    try:
        df3 = pd.read_csv(file_path_d3, encoding='latin-1') # Assuming latin-1 encoding for D3 as well
        df3['dataset_source'] = 'D3'
        print(f"Dataset 3 loaded: {len(df3)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 3 not found at {file_path_d3}. Please ensure it exists.")
        return None

    # --- 1. Harmonize Labels and Rename Columns ---
    print("Harmonizing labels and renaming columns...")
    unified_df1 = df1.copy()
    unified_df1[TARGET_FEATURE] = unified_df1['v1'].apply(lambda x: 1 if x.lower() == 'spam' else 0)
    unified_df1 = unified_df1.rename(columns={'v2': 'sms_content'})[['sms_content', TARGET_FEATURE, 'dataset_source']]

    unified_df2 = df2.copy()
    unified_df2[TARGET_FEATURE] = unified_df2['LABEL'].apply(lambda x: 1 if x.upper() == 'SMISHING' else 0)
    unified_df2 = unified_df2.rename(columns={'TEXT': 'sms_content'})[['sms_content', TARGET_FEATURE, 'dataset_source', 'URL', 'EMAIL', 'PHONE']]

    unified_df3 = df3.copy()
    unified_df3[TARGET_FEATURE] = unified_df3['Phishing'].apply(lambda x: 1 if x > 0 else 0)
    unified_df3 = unified_df3.rename(columns={'MainText': 'sms_content'})

    # Define all possible columns to be generated in the final DataFrame
    all_possible_cols = [
        'sms_content', TARGET_FEATURE, 'dataset_source',
        'has_url', 'has_email', 'has_phone_number',
        'url_string', 'domain', 'tld', 'subdomain', 'is_ip_address_url', 'redirected_url', 'url_subcategory', 'domain_age_days', 'domain_registrar',
        'sender', 'sender_type', 'brand', 'message_category', 'time_received',
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words', 'phishing_keywords_count',
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'word_count', 'avg_word_length',
        'digits_to_length_ratio', 'special_chars_to_length_ratio', 'phishing_keywords_to_word_count_ratio',
        'url_and_ip', 'url_and_suspicious_tld', 'urgent_and_url'
    ]

    # Pre-process df3 to align with final schema and avoid column mismatches during concat
    df3_processed = unified_df3.copy()
    df3_processed['url_string'] = df3_processed['Url']
    df3_processed['redirected_url'] = df3_processed['RedirectedURL']
    df3_processed['url_subcategory'] = df3_processed['URL Subcategory']
    df3_processed['domain_registrar'] = df3_processed['Domain Registrar']
    df3_processed['sender_type'] = df3_processed['SenderType']
    df3_processed['message_category'] = df3_processed['Message Categories']
    df3_processed['time_received'] = pd.to_datetime(df3_processed['timeReceived'], errors='coerce')

    # --- 2. Feature Engineering & Unification ---
    print("Performing feature engineering and unification...")

    # Apply textual heuristics to all DataFrames
    for df in [unified_df1, unified_df2, df3_processed]:
        df[['sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words', 'phishing_keywords_count']] = \
            df['sms_content'].apply(lambda x: pd.Series(extract_textual_heuristics(x, phishing_keywords)))
        
        # Sentiment Features
        df[['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']] = \
            df['sms_content'].apply(lambda x: pd.Series(get_sentiment_scores(x)))
        
        # Readability Proxies
        df['word_count'] = df['sms_content'].apply(lambda x: len(str(x).split()))
        df['avg_word_length'] = df['sms_content'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0)

    # Binary Presence Features (URL, Email, Phone)
    unified_df1['has_url'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'https?://\S+|www\.\S+', str(x))))
    unified_df1['has_email'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'\S+@\S+\.\S+', str(x))))
    unified_df1['has_phone_number'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', str(x))))

    unified_df2['has_url'] = unified_df2['URL'].apply(lambda x: str(x).lower() == 'y')
    unified_df2['has_email'] = unified_df2['EMAIL'].apply(lambda x: str(x).lower() == 'y')
    unified_df2['has_phone_number'] = unified_df2['PHONE'].apply(lambda x: str(x).lower() == 'y')

    df3_processed['has_url'] = df3_processed['url_string'].apply(lambda x: pd.notna(x) and str(x).strip() != '')
    df3_processed['has_email'] = df3_processed['sms_content'].apply(lambda x: bool(re.search(r'\S+@\S+\.\S+', str(x))))
    df3_processed['has_phone_number'] = df3_processed['sms_content'].apply(lambda x: bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', str(x))))

    # Advanced URL Features (from Dataset 3)
    df3_processed[['domain', 'tld', 'subdomain', 'is_ip_address_url']] = \
        df3_processed['Url'].apply(lambda x: pd.Series(extract_url_info(x)))
    
    df3_processed['domain_age_days'] = df3_processed.apply(
        lambda row: calculate_domain_age(row['Domain Creation Date'], row['Domain Last Update']), axis=1
    )

    # Concatenate all processed dataframes to create the unified dataset
    # Reindex all dataframes to ensure they have all possible columns before concatenation
    final_df1 = unified_df1.reindex(columns=all_possible_cols)
    final_df2 = unified_df2.reindex(columns=all_possible_cols)
    final_df3 = df3_processed.reindex(columns=all_possible_cols)

    unified_dataset = pd.concat([final_df1, final_df2, final_df3], ignore_index=True)

    # --- 3. Handling Missing Values (Initial Pass) ---
    print("Handling initial missing values...")

    numerical_cols = [
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
        'phishing_keywords_count', 'domain_age_days',
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'word_count', 'avg_word_length'
    ]
    for col in numerical_cols:
        unified_dataset[col] = unified_dataset[col].fillna(0)

    boolean_cols = ['has_url', 'has_email', 'has_phone_number', 'is_ip_address_url']
    for col in boolean_cols:
        unified_dataset[col] = unified_dataset[col].fillna(False)

    categorical_text_cols = [
        'url_string', 'domain', 'tld', 'subdomain', 'redirected_url', 'url_subcategory', 'domain_registrar',
        'sender', 'sender_type', 'brand', 'message_category'
    ]
    for col in categorical_text_cols:
        unified_dataset[col] = unified_dataset[col].fillna('unknown').replace('', 'unknown')

    if 'time_received' in unified_dataset.columns:
        unified_dataset = unified_dataset.drop(columns=['time_received'])

    # --- 4. Final Text Cleaning (for NLP model input) ---
    unified_dataset[TEXT_FEATURE] = unified_dataset['sms_content'].apply(clean_and_normalize_text_for_nlp)

    # --- Advanced Feature Engineering (Ratio and Interaction Features) ---
    print("Adding ratio and interaction features...")
    unified_dataset['digits_to_length_ratio'] = unified_dataset['num_digits'] / (unified_dataset['sms_length'] + 1e-6)
    unified_dataset['special_chars_to_length_ratio'] = unified_dataset['num_special_chars'] / (unified_dataset['sms_length'] + 1e-6)
    unified_dataset['phishing_keywords_to_word_count_ratio'] = unified_dataset['phishing_keywords_count'] / (unified_dataset['word_count'] + 1e-6)
    
    unified_dataset['url_and_ip'] = unified_dataset['has_url'] & unified_dataset['is_ip_address_url']
    unified_dataset['url_and_suspicious_tld'] = unified_dataset['has_url'] & unified_dataset['tld'].astype(str).isin(['ru', 'cn', 'xyz', 'top', 'loan', 'biz', 'info', 'online'])
    unified_dataset['urgent_and_url'] = (unified_dataset[TEXT_FEATURE].str.contains('urgent|immediately|action', regex=True, na=False)) & unified_dataset['has_url']

    # Categorical Feature Encoding
    print("Encoding categorical features...")
    for col in [f.replace('_encoded', '') for f in CATEGORICAL_FEATURES_FOR_OHE]:
        if col in unified_dataset.columns:
            unified_dataset[col] = unified_dataset[col].fillna('unknown').astype(str)
            le = LabelEncoder()
            unified_dataset[f'{col}_encoded'] = le.fit_transform(unified_dataset[col])
        else:
            print(f"Warning: Original categorical column '{col}' not found for encoding. Creating dummy encoded column.")
            unified_dataset[f'{col}_encoded'] = 0 # Create a dummy encoded column if the original was missing

    print("\nUnified dataset created and features engineered successfully!")
    print(f"Total rows: {len(unified_dataset)}")
    print("Columns and their non-null counts:")
    unified_dataset.info()
    print("\nSample of the unified dataset:")
    print(unified_dataset.head())
    print("\nValue counts for 'is_phishing':")
    print(unified_dataset['is_phishing'].value_counts())

    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Save the processed dataset to checkpoint
    unified_dataset.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to '{output_path}'")

    return unified_dataset

def prepare_data_for_model_input(unified_df):
    """
    Splits and preprocesses the unified DataFrame into model-ready tensors.
    """
    print("\n--- Preparing Data for Model Input ---")

    X = unified_df.drop(columns=[TARGET_FEATURE])
    y = unified_df[TARGET_FEATURE]

    # Use 80% for training+validation, 20% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Split train_val into 75% train and 25% val (0.25 of 0.8 is 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    # Tokenize text features
    train_input_ids, train_attention_mask = tokenize_text(X_train[TEXT_FEATURE], tokenizer, MAX_LEN)
    val_input_ids, val_attention_mask = tokenize_text(X_val[TEXT_FEATURE], tokenizer, MAX_LEN)
    test_input_ids, test_attention_mask = tokenize_text(X_test[TEXT_FEATURE], tokenizer, MAX_LEN)
    
    # Filter features to ensure they exist in the DataFrame
    numerical_features_for_scaling = [col for col in NUMERICAL_FEATURES_FOR_SCALING if col in X_train.columns]
    boolean_features_as_numerical = [col for col in BOOLEAN_FEATURES_AS_NUMERICAL if col in X_train.columns]
    categorical_features_for_ohe = [col for col in CATEGORICAL_FEATURES_FOR_OHE if col in X_train.columns]

    all_numerical_cols_for_scaling = numerical_features_for_scaling + boolean_features_as_numerical
    for col in all_numerical_cols_for_scaling:
        X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_val.loc[:, col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
        X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    # Structured features preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_for_scaling + boolean_features_as_numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_ohe)
        ],
        remainder='drop'
    )
    X_train_structured_processed = preprocessor.fit_transform(X_train)
    X_val_structured_processed = preprocessor.transform(X_val)
    X_test_structured_processed = preprocessor.transform(X_test)

    # Convert structured features to dense TensorFlow tensors if they are sparse
    if hasattr(X_train_structured_processed, 'toarray'):
        X_train_structured_processed = tf.constant(X_train_structured_processed.toarray(), dtype=tf.float32)
        X_val_structured_processed = tf.constant(X_val_structured_processed.toarray(), dtype=tf.float32)
        X_test_structured_processed = tf.constant(X_test_structured_processed.toarray(), dtype=tf.float32)
    else:
        X_train_structured_processed = tf.constant(X_train_structured_processed, dtype=tf.float32)
        X_val_structured_processed = tf.constant(X_val_structured_processed, dtype=tf.float32)
        X_test_structured_processed = tf.constant(X_test_structured_processed, dtype=tf.float32)

    # Convert labels to TensorFlow tensors
    y_train = tf.constant(y_train.values, dtype=tf.float32)
    y_val = tf.constant(y_val.values, dtype=tf.float32)
    y_test = tf.constant(y_test.values, dtype=tf.float32)

    print("Data preparation for model input complete.")
    return ProcessedData(
        train_input_ids=train_input_ids,
        train_attention_mask=train_attention_mask,
        X_train_structured_processed=X_train_structured_processed,
        y_train=y_train,
        val_input_ids=val_input_ids,
        val_attention_mask=val_attention_mask,
        X_val_structured_processed=X_val_structured_processed,
        y_val=y_val,
        test_input_ids=test_input_ids,
        test_attention_mask=test_attention_mask,
        X_test_structured_processed=X_test_structured_processed,
        y_test=y_test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        X_train_original_df=X_train,
        X_test_original_df=X_test
    )

# --- Plotting Functions (moved from data_processing.py) ---
def plot_training_history(history, save_path=None):
    """
    Plots training and validation accuracy and loss curves from a Keras History object.
    Saves the plot if save_path is provided.
    """
    hist = history.history

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in hist:
        plt.plot(hist['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Train Loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred_class, save_path=None):
    """
    Plots the confusion matrix.
    Saves the plot if save_path is provided.
    """
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Ham', 'Predicted Phishing'],
                yticklabels=['Actual Ham', 'Actual Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plots the ROC curve and displays AUC.
    Saves the plot if save_path is provided.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to {save_path}")
    plt.show()
    plt.close()
    return roc_auc

def plot_adversarial_accuracy_history(clean_acc_history, adv_acc_history, save_path=None):
    """
    Plots the history of clean and adversarial test accuracies over iterations.
    Saves the plot if save_path is provided.
    """
    plt.figure(figsize=(10, 6))
    
    # Adjust x-axis for clean accuracy to match iterations correctly
    # Clean accuracy history has initial clean, then clean after iter 1, clean after iter 2...
    # Adv accuracy history has initial 0.0, then adv after iter 1, adv after iter 2...
    
    # X-axis points for iterations: 0 (initial), 1, 2, ... NUM_ADVERSARIAL_ITERATIONS
    # Clean history has 2*NUM_ADVERSARIAL_ITERATIONS + 1 points (initial, then 2 per iter)
    # Adv history has NUM_ADVERSARIAL_ITERATIONS + 1 points (initial 0.0, then 1 per iter)
    
    # Plot clean accuracy at the start of each iteration (or initial)
    plt.plot(np.arange(len(clean_acc_history)), clean_acc_history, 'o-', label='Clean Test Accuracy', color='blue')
    
    # Plot adversarial accuracy at the end of each iteration
    # Shift x-axis by 0.5 to center on the iteration number
    plt.plot(np.arange(len(adv_acc_history)), adv_acc_history, 's-', label='Adversarial Test Accuracy', color='red')
    
    plt.title('Model Accuracy History: Clean vs. Adversarial')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(len(adv_acc_history))) # Set x-ticks to correspond to iterations
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Adversarial accuracy history plot saved to {save_path}")
    plt.show()
    plt.close()
