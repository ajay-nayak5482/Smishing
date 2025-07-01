import pandas as pd
import re
import numpy as np
from datetime import datetime
import tldextract
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import warnings

# --- Configuration ---
# File paths for your datasets
NAME_DATASET1 = 'dataset1.csv'
NAME_DATASET2 = 'dataset2.csv'
NAME_DATASET3 = 'dataset3.csv'
STOP_WORDS = set()
analyzer = None
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output
# Define common phishing/spam keywords (expand this list significantly for real project)
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
    'verification code', 'authentication required', 'unauthorized access attempt',
    'malicious link', 'suspicious website', 'virus alert', 'ransomware attack',
    'exploit detected', 'vulnerability found', 'data leak', 'breach notification'
]

# --- Helper Functions for Feature Engineering ---

def extract_url_info(url_string):
    """Extracts domain, TLD, and subdomain using tldextract."""
    if pd.isna(url_string) or not isinstance(url_string, str) or not url_string.strip():
        return None, None, None, False # No URL string or empty
    
    # Prepend http if missing to help tldextract parse (it's robust but sometimes helps)
    if not re.match(r'^[a-zA-Z]+://', url_string):
        url_string = 'http://' + url_string

    try:
        extracted = tldextract.extract(url_string)
        # Check if it's an IP address instead of a domain name
        is_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', extracted.domain))
        
        # tldextract returns empty string for non-existent parts, convert to None
        subdomain = extracted.subdomain if extracted.subdomain else None
        domain = extracted.domain if extracted.domain else None
        tld = extracted.suffix if extracted.suffix else None # suffix is the TLD
        
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
    
    # Use the earliest of the two dates to calculate age from today
    earliest_date = min(dates_to_consider)
    return (datetime.now() - earliest_date).days

def clean_text(text):
    """Basic text cleaning for NLP features."""
    if pd.isna(text):
        return ""
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text) # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def extract_textual_heuristics(text):
    """Extracts various heuristic features from text."""
    if pd.isna(text):
        return 0, 0, 0, 0, 0
    
    text_cleaned = clean_text(text) # Use the cleaned text for most heuristics

    sms_length = len(text_cleaned)
    num_special_chars = sum(1 for char in text if char in string.punctuation) # Use original text for this
    num_digits = sum(1 for char in text if char.isdigit())
    
    words = word_tokenize(text_cleaned)
    num_all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    phishing_keywords_count = sum(1 for keyword in PHISHING_KEYWORDS if keyword in text_cleaned)
    
    return sms_length, num_special_chars, num_digits, num_all_caps_words, phishing_keywords_count

# --- Enhanced Text Cleaning Function ---
# This version is more explicit about what it removes and can be used for initial text prep.
# For transformer input, you'll generally use the tokenizer's specific preprocessing.
# For heuristic features, this level of cleaning is good.



def clean_and_normalize_text_for_heuristics(text):
    """
    Performs comprehensive cleaning and normalization for heuristic feature extraction.
    This version is more aggressive than the previous 'clean_text' as it removes stopwords.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower() # Convert to string and lowercase

    # Remove URLs (keep this before other cleaning if you want to detect has_url first)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # Remove text in square brackets (e.g., [image])
    text = re.sub(r'\[.*?\]', '', text) 

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(words)

def get_sentiment_scores(text):
    """Calculates VADER sentiment scores."""
    if pd.isna(text) or not text.strip():
        return 0.0, 0.0, 0.0, 0.0 # neg, neu, pos, compound
    scores = analyzer.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

# --- Main Data Integration Function ---

def integrate_datasets(file_path_d1, file_path_d2, file_path_d3):
    """
    Loads, preprocesses, and integrates all three datasets into a single DataFrame.
    """
    print("Loading datasets...")
    # Load Dataset 1
    try:
        df1 = pd.read_csv(file_path_d1, encoding='latin-1')
        df1['dataset_source'] = 'D1'
        print(f"Dataset 1 loaded: {len(df1)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 1 not found at {file_path_d1}. Creating dummy data.")
       
    # Load Dataset 2
    try:
        df2 = pd.read_csv(file_path_d2)
        df2['dataset_source'] = 'D2'
        print(f"Dataset 2 loaded: {len(df2)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 2 not found at {file_path_d2}. Creating dummy data.")
        

    # Load Dataset 3
    try:
        df3 = pd.read_csv(file_path_d3, encoding='latin-1')
        df3['dataset_source'] = 'D3'
        print(f"Dataset 3 loaded: {len(df3)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset 3 not found at {file_path_d3}. Creating dummy data.")
        


    # --- 1. Harmonize Labels ---
    print("Harmonizing labels...")
    unified_df1 = df1.copy()
    unified_df1['is_phishing'] = unified_df1['v1'].apply(lambda x: 1 if x.lower() == 'spam' else 0)
    unified_df1 = unified_df1.rename(columns={'v2': 'sms_content'})[['sms_content', 'is_phishing', 'dataset_source']]

    unified_df2 = df2.copy()
    unified_df2['is_phishing'] = unified_df2['LABEL'].apply(lambda x: 1 if x.upper() == 'SMISHING' else 0)
    unified_df2 = unified_df2.rename(columns={'TEXT': 'sms_content'})[['sms_content', 'is_phishing', 'dataset_source', 'URL', 'EMAIL', 'PHONE']]

    unified_df3 = df3.copy()
    unified_df3['is_phishing'] = unified_df3['Phishing'].apply(lambda x: 1 if x > 0 else 0)
    unified_df3 = unified_df3.rename(columns={'MainText': 'sms_content'})

    # Select and rename columns for DF3 to match a target schema, handling missing ones
    # Define all possible columns to be generated in the final DataFrame
    all_possible_cols = [
        'sms_content', 'is_phishing', 'dataset_source',
        'has_url', 'has_email', 'has_phone_number',
        'url_string', 'domain', 'tld', 'subdomain', 'is_ip_address_url', 'redirected_url', 'url_subcategory', 'domain_age_days', 'domain_registrar',
        'sender', 'sender_type', 'brand', 'message_category', 'time_received',
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words', 'phishing_keywords_count'
    ]

    # Pre-process df3 to align with final schema and avoid column mismatches during concat
    df3_processed = unified_df3.copy()
    df3_processed['url_string'] = df3_processed['Url'] # Keep original URL field
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
            df['sms_content'].apply(lambda x: pd.Series(extract_textual_heuristics(x)))

    # Binary Presence Features (URL, Email, Phone)
    # For D1, derive using regex on sms_content
    unified_df1['has_url'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'https?://\S+|www\.\S+', str(x))))
    unified_df1['has_email'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'\S+@\S+\.\S+', str(x))))
    unified_df1['has_phone_number'] = unified_df1['sms_content'].apply(lambda x: bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', str(x))))

    # For D2, use existing columns and convert 'y'/'n' to boolean
    unified_df2['has_url'] = unified_df2['URL'].apply(lambda x: x.lower() == 'y')
    unified_df2['has_email'] = unified_df2['EMAIL'].apply(lambda x: x.lower() == 'y')
    unified_df2['has_phone_number'] = unified_df2['PHONE'].apply(lambda x: x.lower() == 'y')

    # For D3, derive from `Url` column or `sms_content` if Url is missing
    df3_processed['has_url'] = df3_processed['url_string'].apply(lambda x: pd.notna(x) and str(x).strip() != '')
    df3_processed['has_email'] = df3_processed['sms_content'].apply(lambda x: bool(re.search(r'\S+@\S+\.\S+', str(x))))
    df3_processed['has_phone_number'] = df3_processed['sms_content'].apply(lambda x: bool(re.search(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', str(x))))

    # Advanced URL Features (from Dataset 3)
    df3_processed[['domain', 'tld', 'subdomain', 'is_ip_address_url']] = \
        df3_processed['Url'].apply(lambda x: pd.Series(extract_url_info(x)))
    
    # Calculate domain age for Dataset 3
    df3_processed['domain_age_days'] = df3_processed.apply(
        lambda row: calculate_domain_age(row['Domain Creation Date'], row['Domain Last Update']), axis=1
    )

    # Consolidate column selection and concatenation
    # Ensure all dataframes have all `all_possible_cols` before concat, filling missing with NaN
    final_df1 = unified_df1.reindex(columns=all_possible_cols)
    final_df2 = unified_df2.reindex(columns=all_possible_cols)
    final_df3 = df3_processed.reindex(columns=all_possible_cols) # This should have most of the D3 features

    # Concatenate all processed dataframes
    unified_dataset = pd.concat([final_df1, final_df2, final_df3], ignore_index=True)

    # --- 3. Handling Missing Values ---
    print("Handling missing values...")

    # Fill numerical columns with 0 or a reasonable default
    numerical_cols = [
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
        'phishing_keywords_count', 'domain_age_days'
    ]
    for col in numerical_cols:
        unified_dataset[col] = unified_dataset[col].fillna(0)

    # Fill boolean columns with False
    boolean_cols = ['has_url', 'has_email', 'has_phone_number', 'is_ip_address_url']
    for col in boolean_cols:
        unified_dataset[col] = unified_dataset[col].fillna(False)

    # Fill categorical/text columns with 'unknown' or ''
    categorical_text_cols = [
        'url_string', 'domain', 'tld', 'subdomain', 'redirected_url', 'url_subcategory', 'domain_registrar',
        'sender', 'sender_type', 'brand', 'message_category'
    ]
    for col in categorical_text_cols:
        unified_dataset[col] = unified_dataset[col].fillna('unknown').replace('', 'unknown') # Also handle empty strings

    # Drop time_received for now as it's complex for simple fillna and might need specific handling
    if 'time_received' in unified_dataset.columns:
        unified_dataset = unified_dataset.drop(columns=['time_received'])

    # # --- 4. Final Text Cleaning (for NLP model input) ---
    # print("Applying final text cleaning...")
    # unified_dataset['sms_content_cleaned'] = unified_dataset['sms_content'].apply(clean_text)

    # print("\nUnified dataset created successfully!")
    # print(f"Total rows: {len(unified_dataset)}")
    # print("Columns and their non-null counts:")
    # print(unified_dataset.info())
    # print("\nSample of the unified dataset:")
    # print(unified_dataset.head())
    # print("\nValue counts for 'is_phishing':")
    # print(unified_dataset['is_phishing'].value_counts())

    # return unified_dataset

        # --- 4. Final Text Cleaning (for NLP model input) ---
    # This column will be the primary text input for the transformer model
    unified_dataset['sms_content_cleaned_for_nlp'] = unified_dataset['sms_content'].apply(clean_and_normalize_text_for_heuristics)

    return unified_dataset

# --- Execute the integration ---
def preprocess_datasets(working_dir):
    """Preprocesses the datasets by integrating them into a unified DataFrame."""
    print("Starting dataset integration...")
    data_dir = os.path.join(working_dir, '..\data')
    data_dir = os.path.abspath(data_dir)  # Ensure absolute path
    
    path_dataset1 = os.path.join(data_dir, NAME_DATASET1)
    path_dataset2 = os.path.join(data_dir, NAME_DATASET2)
    path_dataset3 = os.path.join(data_dir, NAME_DATASET3)
    print(f"File paths set:\nD1: {path_dataset1}\nD2: {path_dataset2}\nD3: {path_dataset3}")

    unified_df = integrate_datasets(path_dataset1, path_dataset2, path_dataset3)

    # Example of how you might save the unified dataset
    unified_df.to_csv(os.path.join(data_dir,'unified_phishing_sms_dataset.csv'), index=False)
    print("\nUnified dataset saved to 'unified_phishing_sms_dataset.csv'")

    # You can now proceed with your model training using `unified_df`
    # For example:
    # X_text = unified_df['sms_content_cleaned']
    # X_structured = unified_df[['has_url', 'has_email', ..., 'phishing_keywords_count']]
    # y = unified_df['is_phishing']
    return unified_df

def clean_and_prepare_datasets(unified_df):
    print("\n--- Starting Data Cleaning, Exploration, and Advanced Feature Engineering ---")

    # --- 1. Data Cleaning & Initial Exploration ---

    print("\n--- Initial Data Overview ---")
    print(f"Shape of unified dataset: {unified_df.shape}")
    print("\nMissing values after initial handling:")
    print(unified_df.isnull().sum()[unified_df.isnull().sum() > 0]) # Should be mostly zeros now

    print("\nDuplicate rows before removal:")
    print(unified_df.duplicated().sum())
    unified_df.drop_duplicates(inplace=True)
    print(f"Duplicate rows after removal: {unified_df.duplicated().sum()}")
    print(f"Shape after duplicate removal: {unified_df.shape}")

    print("\nClass distribution ('is_phishing'):")
    print(unified_df['is_phishing'].value_counts(normalize=True))

    print("\nSummary statistics for numerical features:")
    print(unified_df.describe())

    # --- 2. Advanced Feature Engineering ---

    print("\n--- Performing Advanced Feature Engineering ---")

    # Sentiment Features (from sms_content_cleaned_for_nlp for better accuracy)
    unified_df[['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']] = \
        unified_df['sms_content_cleaned_for_nlp'].apply(lambda x: pd.Series(get_sentiment_scores(x)))
    print("Added sentiment features.")

    # Readability Score (using a simple approximation, for a full one, you'd need textstat library)
    # For simplicity, let's use word count / sentence count (rough proxy)
    unified_df['word_count'] = unified_df['sms_content_cleaned_for_nlp'].apply(lambda x: len(x.split()))
    unified_df['avg_word_length'] = unified_df['sms_content_cleaned_for_nlp'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
    print("Added word count and average word length.")

    # Ratio Features
    unified_df['digits_to_length_ratio'] = unified_df['num_digits'] / (unified_df['sms_length'] + 1e-6)
    unified_df['special_chars_to_length_ratio'] = unified_df['num_special_chars'] / (unified_df['sms_length'] + 1e-6)
    unified_df['phishing_keywords_to_word_count_ratio'] = unified_df['phishing_keywords_count'] / (unified_df['word_count'] + 1e-6)
    print("Added ratio features.")

    # Interaction Features (examples)
    unified_df['url_and_ip'] = unified_df['has_url'] & unified_df['is_ip_address_url']
    unified_df['url_and_suspicious_tld'] = unified_df['has_url'] & unified_df['tld'].isin(['ru', 'cn', 'xyz', 'top', 'loan', 'biz', 'info', 'online']) # Expand suspicious TLDs
    unified_df['urgent_and_url'] = (unified_df['sms_content_cleaned_for_nlp'].str.contains('urgent|immediately|action', regex=True)) & unified_df['has_url']
    print("Added interaction features.")

    # Categorical Feature Encoding (for structured model input)
    # For tree-based models, label encoding might be sufficient. For NNs, one-hot or embeddings.
    # Let's do Label Encoding for simplicity here, but keep in mind for final model.
    
    categorical_features_to_encode = [
        'domain', 'tld', 'subdomain', 'url_subcategory', 'domain_registrar',
        'sender', 'sender_type', 'brand', 'message_category', 'dataset_source'
    ]

    for col in categorical_features_to_encode:
        # Ensure 'unknown' is treated as a category, and fillna again just in case
        unified_df[col] = unified_df[col].fillna('unknown').astype(str)
        le = LabelEncoder()
        unified_df[f'{col}_encoded'] = le.fit_transform(unified_df[col])
        print(f"Encoded '{col}' to '{col}_encoded'.")

    print("\n--- Feature Engineering Complete ---")
    print(f"New shape of dataset: {unified_df.shape}")
    print("Sample of dataset with new features:")
    print(unified_df.head())

    # --- 3. Exploratory Data Analysis (EDA) ---

    print("\n--- Starting Exploratory Data Analysis (EDA) ---")

    # Distribution of SMS Length
    plt.figure(figsize=(10, 6))
    sns.histplot(data=unified_df, x='sms_length', hue='is_phishing', kde=True, palette='viridis')
    plt.title('Distribution of SMS Length by Phishing Status')
    plt.xlabel('SMS Length (characters)')
    plt.ylabel('Count')
    plt.show()

    # Distribution of Phishing Keywords Count
    plt.figure(figsize=(10, 6))
    sns.histplot(data=unified_df, x='phishing_keywords_count', hue='is_phishing', kde=True, palette='magma')
    plt.title('Distribution of Phishing Keywords Count by Phishing Status')
    plt.xlabel('Phishing Keywords Count')
    plt.ylabel('Count')
    plt.show()

    # Bar plots for binary/categorical features vs. Phishing Status
    binary_features = ['has_url', 'has_email', 'has_phone_number', 'is_ip_address_url', 'url_and_ip', 'urgent_and_url']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(binary_features):
        plt.subplot(2, 3, i + 1)
        sns.countplot(data=unified_df, x=feature, hue='is_phishing', palette='cividis')
        plt.title(f'Phishing Status by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Sentiment Score Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=unified_df, x='sentiment_compound', hue='is_phishing', kde=True, palette='plasma')
    plt.title('Distribution of Compound Sentiment Score by Phishing Status')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Count')
    plt.show()

    # Word Clouds for Phishing vs. Ham
    phishing_text = " ".join(unified_df[unified_df['is_phishing'] == 1]['sms_content_cleaned_for_nlp'].dropna())
    ham_text = " ".join(unified_df[unified_df['is_phishing'] == 0]['sms_content_cleaned_for_nlp'].dropna())

    if phishing_text:
        phishing_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(phishing_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(phishing_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Phishing SMS')
        plt.show()
    else:
        print("Not enough phishing text to generate word cloud.")

    if ham_text:
        ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(ham_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Ham SMS')
        plt.show()
    else:
        print("Not enough ham text to generate word cloud.")

    # Correlation Matrix of Numerical Features
    numerical_features_for_corr = [
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
        'phishing_keywords_count', 'domain_age_days',
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'word_count', 'avg_word_length',
        'digits_to_length_ratio', 'special_chars_to_length_ratio', 'phishing_keywords_to_word_count_ratio',
        'is_phishing' # Include target for correlation with features
    ]
    
    # Filter to only existing columns
    numerical_features_for_corr = [col for col in numerical_features_for_corr if col in unified_df.columns]

    plt.figure(figsize=(12, 10))
    sns.heatmap(unified_df[numerical_features_for_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

    print("\n--- EDA Complete ---")

    # --- Save the processed dataset ---
    output_file_name = 'unified_phishing_sms_dataset_processed.csv'
    unified_df.to_csv(output_file_name, index=False)
    print(f"\nProcessed dataset saved to '{output_file_name}'")



if __name__ == "__main__":
    import os
    import sys
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Ensure the script is run with Python 3
    if sys.version_info[0] < 3:
        print("This script requires Python 3.")
        sys.exit(1)

    # Set the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    # Import and run the main functionality of the application
    # from app import main_functionality
    # main_functionality.run()
    # Download NLTK data if not already present
    global analyzer    
    try:
        stopwords.words('english')
        word_tokenize("test")
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        analyzer.polarity_scores("test")
    except LookupError:
        print("Downloading NLTK data (stopwords, punkt)...")        
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        print("NLTK data downloaded.")
    
    global STOP_WORDS
    STOP_WORDS = set(stopwords.words('english'))
    if(analyzer is None):
        analyzer = SentimentIntensityAnalyzer()
    unified_df = preprocess_datasets(script_dir)
    clean_and_prepare_datasets(unified_df)