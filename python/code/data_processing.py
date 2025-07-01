import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer # For transformer text preprocessing
import numpy as np
import warnings
from matplotlib import pyplot as plt
from model_config import ModelTrainingParams # Assuming this is where your model params are defined
import model_training # Assuming this is where your model training function is defined

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

def tokenize_text(texts, tokenizer, max_len):
        """
        Tokenizes a list of texts for transformer input.
        Returns input_ids, attention_mask, token_type_ids (if applicable).
        """
        # Ensure all texts are strings and handle potential NaN/None values
        texts = texts.astype(str).fillna('').tolist()
        
        encodings = tokenizer.batch_encode_plus(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='np'
        )
        return encodings['input_ids'], encodings['attention_mask'], encodings.get('token_type_ids', np.zeros(encodings['input_ids'].shape))

def process_data_for_training():
    try:
        unified_df = pd.read_csv('unified_phishing_sms_dataset_processed.csv')
        print("Loaded processed unified dataset.")
    except FileNotFoundError:
        print("Processed dataset CSV not found. Please run the previous data integration and EDA script first.")
        # If you need to re-run the full integration from scratch:
        # from your_previous_script_name import integrate_datasets # Assuming you saved it as a module
        # unified_df = integrate_datasets(FILE_PATH_D1, FILE_PATH_D2, FILE_PATH_D3)
        exit() # Exit if the dataframe is not available

    # --- Configuration for Model Building ---
    # Choose a transformer model. DistilBERT is a good lightweight choice for mobile deployment.
    TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
    MAX_LEN = 128 # Maximum sequence length for transformer input

    # --- 1. Data Splitting ---
    print("\n--- Splitting Data into Train, Validation, and Test Sets ---")

    # Define features and target
    # 'sms_content_cleaned_for_nlp' will be for the transformer branch
    # Other columns (excluding original text and source info) for the structured branch
    text_feature = 'sms_content_cleaned_for_nlp'
    target_feature = 'is_phishing'

    # Identify structured features (excluding text, target, and original source/raw text columns)
    # We'll also exclude the original categorical columns, only keeping their encoded versions
    structured_features = [col for col in unified_df.columns if col not in [
        'sms_content', 'sms_content_cleaned_for_nlp', 'is_phishing', 'dataset_source',
        'url_string', 'domain', 'tld', 'subdomain', 'redirected_url', 'url_subcategory',
        'domain_registrar', 'sender', 'sender_type', 'brand', 'message_category'
    ]]

    # Separate features (X) and target (y)
    X = unified_df.drop(columns=[target_feature])
    y = unified_df[target_feature]

    # Stratified split to maintain class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


    # --- 2. Text Preprocessing for Transformer ---
    print(f"\n--- Initializing Tokenizer for {TRANSFORMER_MODEL_NAME} ---")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    

    # Apply tokenization to training, validation, and test sets
    print("Tokenizing text data for train, validation, and test sets...")
    train_input_ids, train_attention_mask, train_token_type_ids = tokenize_text(X_train[text_feature], tokenizer, MAX_LEN)
    val_input_ids, val_attention_mask, val_token_type_ids = tokenize_text(X_val[text_feature], tokenizer, MAX_LEN)
    test_input_ids, test_attention_mask, test_token_type_ids = tokenize_text(X_test[text_feature], tokenizer, MAX_LEN)

    print(f"Shape of train_input_ids: {train_input_ids.shape}")
    print(f"Shape of train_attention_mask: {train_attention_mask.shape}")


    # --- 3. Structured Feature Preprocessing ---
    print("\n--- Preprocessing Structured Features ---")

    # Re-define structured features more precisely for preprocessing
    # Numerical features (already cleaned/filled with 0/False in previous step)
    numerical_features_for_scaling = [
        'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
        'phishing_keywords_count', 'domain_age_days',
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'word_count', 'avg_word_length',
        'digits_to_length_ratio', 'special_chars_to_length_ratio', 'phishing_keywords_to_word_count_ratio'
    ]

    # Boolean features (already 0/1, can be treated as numerical or left as is)
    boolean_features_as_numerical = [
        'has_url', 'has_email', 'has_phone_number', 'is_ip_address_url',
        'url_and_ip', 'url_and_suspicious_tld', 'urgent_and_url'
    ]

    # Categorical features (using the '_encoded' versions)
    categorical_features_for_ohe = [
        'domain_encoded', 'tld_encoded', 'subdomain_encoded', 'url_subcategory_encoded',
        'domain_registrar_encoded', 'sender_encoded', 'sender_type_encoded',
        'brand_encoded', 'message_category_encoded', 'dataset_source_encoded'
    ]

    # Filter to only include columns that actually exist in the DataFrame
    numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in X_train.columns]
    boolean_features_as_numerical = [col for col in boolean_features_as_numerical if col in X_train.columns]
    categorical_features_for_ohe = [col for col in categorical_features_for_ohe if col in X_train.columns]

    # --- FIX: Ensure all numerical/boolean columns are explicitly numeric (float) ---
    # Combine all columns that will go into the StandardScaler
    all_numerical_cols_for_scaling = numerical_features_for_scaling + boolean_features_as_numerical

    for col in all_numerical_cols_for_scaling:
        # Convert to numeric, coercing errors (non-numeric strings become NaN)
        X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce')
        X_val.loc[:, col] = pd.to_numeric(X_val[col], errors='coerce')
        X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Fill any NaNs that resulted from coercion (or were already there) with 0
        X_train.loc[:, col] = X_train[col].fillna(0)
        X_val.loc[:, col] = X_val[col].fillna(0)
        X_test.loc[:, col] = X_test[col].fillna(0)
    print("Ensured all numerical and boolean features are converted to numeric (float) and NaNs filled.")


    # Create a preprocessor using ColumnTransformer
    # Numerical features will be scaled
    # Categorical features will be one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_for_scaling + boolean_features_as_numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_ohe)
        ],
        remainder='drop' # Drop any columns not specified
    )

    # Fit and transform structured features
    print("Fitting preprocessor and transforming structured features for train, validation, and test sets...")
    X_train_structured_processed = preprocessor.fit_transform(X_train)
    X_val_structured_processed = preprocessor.transform(X_val)
    X_test_structured_processed = preprocessor.transform(X_test)

    print(f"Shape of processed train structured features: {X_train_structured_processed.shape}")
    print(f"Shape of processed val structured features: {X_val_structured_processed.shape}")
    print(f"Shape of processed test structured features: {X_test_structured_processed.shape}")
    model_training_params = ModelTrainingParams(
        X_train_structured_processed=X_train_structured_processed,
        X_val_structured_processed=X_val_structured_processed,
        X_test_structured_processed=X_test_structured_processed,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        train_input_ids=train_input_ids,
        train_attention_mask=train_attention_mask,
        train_token_type_ids=train_token_type_ids,
        val_input_ids=val_input_ids,
        val_attention_mask=val_attention_mask,
        val_token_type_ids=val_token_type_ids,
        test_input_ids=test_input_ids,
        test_attention_mask=test_attention_mask,
        test_token_type_ids=test_token_type_ids
    )
    return model_training_params
# Next step: Define and train your hybrid model using these prepared inputs.
# This will involve using a deep learning framework like TensorFlow/Keras or PyTorch.


def plot_training_history(history):
        """
        Plots training and validation accuracy and loss curves from a Keras History object.
        """
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history

        plt.figure(figsize=(12, 5))

        # Plot accuracy
        if 'accuracy' in hist or 'acc' in hist:
            acc_key = 'accuracy' if 'accuracy' in hist else 'acc'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else 'val_acc'
            plt.subplot(1, 2, 1)
            plt.plot(hist[acc_key], label='Train Accuracy')
            if val_acc_key in hist:
                plt.plot(hist[val_acc_key], label='Val Accuracy')
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
        plt.show()
        
if __name__ == "__main__":
    model_training_params = process_data_for_training()  # Pass an empty DataFrame as we are loading the unified dataset from CSV
    model, history = model_training.define_and_train_model(model_training_params)  # Assuming this is where your model training function is defined
    
    plot_training_history(history)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('model_training_history.csv', index=False)
    print("Data processing script completed successfully.")