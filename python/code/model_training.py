import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda # Import Lambda layer
from tensorflow.keras.models import Model
from transformers import TFDistilBertModel, DistilBertConfig, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- Configuration (from previous script, ensure consistency) ---
TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128 # Maximum sequence length for transformer input
DISTILBERT_HIDDEN_SIZE = 768 # Hidden size for distilbert-base-uncased

@dataclass
class ModelConfig:
    """
    Configuration parameters for building and training the hybrid model.
    """
    transformer_model_name: str
    max_len: int
    learning_rate: float
    epochs: int
    batch_size: int
    
    # Training data
    train_input_ids: tf.Tensor
    train_attention_mask: tf.Tensor
    X_train_structured_processed: tf.Tensor
    y_train: tf.Tensor

    # Validation data
    val_input_ids: tf.Tensor
    val_attention_mask: tf.Tensor
    X_val_structured_processed: tf.Tensor
    y_val: tf.Tensor

    # Test data
    test_input_ids: tf.Tensor
    test_attention_mask: tf.Tensor
    X_test_structured_processed: tf.Tensor
    y_test: tf.Tensor

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
        return_tensors='tf' # Return TensorFlow tensors for Keras model
    )
    return encodings['input_ids'], encodings['attention_mask']


def build_and_train_hybrid_model(config: ModelConfig):
    """
    Builds, compiles, trains, and evaluates the hybrid SMS phishing detection model.

    Args:
        config (ModelConfig): An instance of ModelConfig containing all necessary parameters and data.
    """
    print("\n--- Defining Hybrid Model Architecture ---")

    # Text Branch Input
    input_ids = Input(shape=(config.max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(config.max_len,), dtype=tf.int32, name='attention_mask')

    # Load pre-trained DistilBERT model
    distilbert_model_layer = TFDistilBertModel.from_pretrained(config.transformer_model_name, trainable=True, return_dict=False)

    # --- MODIFICATION: Wrap the DistilBERT model call in a Lambda layer with output_shape ---
    # The Lambda layer helps bridge potential type mismatches between KerasTensors and
    # the internal expectations of some Hugging Face TF models in the functional API.
    def call_distilbert(inputs):
        input_ids_tensor, attention_mask_tensor = inputs
        return distilbert_model_layer(
            input_ids=input_ids_tensor, 
            attention_mask=attention_mask_tensor
        )
    
    # Specify output_shape as a list of shapes, since distilbert_model_layer returns a tuple (last_hidden_state,)
    # The shape of last_hidden_state is (batch_size, sequence_length, hidden_size)
    # So, for output_shape, we provide (sequence_length, hidden_size)
    transformer_outputs = Lambda(
        call_distilbert, 
        output_shape=[(config.max_len, DISTILBERT_HIDDEN_SIZE)], # MODIFIED: Added output_shape
        name='distilbert_wrapper'
    )([input_ids, attention_mask])
    # --- END MODIFICATION ---

    # transformer_outputs will now be a tuple, so access the first element (last_hidden_state)
    text_features = transformer_outputs[0][:, 0, :]

    text_features = Dropout(0.2)(text_features)

    # Structured Features Branch Input
    # Get the number of features from the preprocessed structured data
    structured_input_dim = config.X_train_structured_processed.shape[1]
    structured_features_input = Input(shape=(structured_input_dim,), dtype=tf.float32, name='structured_features_input')

    # Small MLP for structured features
    structured_features_mlp = Dense(128, activation='relu')(structured_features_input)
    structured_features_mlp = Dropout(0.2)(structured_features_mlp)
    structured_features_mlp = Dense(64, activation='relu')(structured_features_mlp)
    structured_features_mlp = Dropout(0.2)(structured_features_mlp)

    # Concatenate Text and Structured Features
    combined_features = Concatenate()([text_features, structured_features_mlp])

    # Final Classification Head
    output = Dense(64, activation='relu')(combined_features)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output) # Sigmoid for binary classification

    # Create the Hybrid Model
    model = Model(inputs=[input_ids, attention_mask, structured_features_input], outputs=output)

    # --- Compile the Model ---
    print("\n--- Compiling Model ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary()

    # --- Handle Class Imbalance ---
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(config.y_train),
        y=config.y_train.numpy()
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"\nCalculated Class Weights: {class_weights_dict}")

    # --- Train the Model ---
    print("\n--- Training Model ---")

    train_inputs = {
        'input_ids': config.train_input_ids,
        'attention_mask': config.train_attention_mask,
        'structured_features_input': config.X_train_structured_processed
    }

    val_inputs = {
        'input_ids': config.val_input_ids,
        'attention_mask': config.val_attention_mask,
        'structured_features_input': config.X_val_structured_processed
    }

    history = model.fit(
        train_inputs,
        config.y_train,
        validation_data=(val_inputs, config.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights_dict
    )

    print("\n--- Model Training Complete ---")

    # --- Evaluate the Model on Test Set ---
    print("\n--- Evaluating Model on Test Set ---")

    test_inputs = {
        'input_ids': config.test_input_ids,
        'attention_mask': config.test_attention_mask,
        'structured_features_input': config.X_test_structured_processed
    }

    loss, accuracy, precision, recall = model.evaluate(test_inputs, config.y_test)

    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    # Return the trained model and history for further use (e.g., saving)
    return model, history


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Prepared Data ---
    try:
        unified_df = pd.read_csv('unified_phishing_sms_dataset_processed.csv')
        print("Loaded processed unified dataset for model building.")

        # Re-run data splitting and preprocessing to get the arrays
        text_feature = 'sms_content_cleaned_for_nlp'
        target_feature = 'is_phishing'

        X = unified_df.drop(columns=[target_feature])
        y = unified_df[target_feature]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

        train_input_ids, train_attention_mask = tokenize_text(X_train[text_feature], tokenizer, MAX_LEN)
        val_input_ids, val_attention_mask = tokenize_text(X_val[text_feature], tokenizer, MAX_LEN)
        test_input_ids, test_attention_mask = tokenize_text(X_test[text_feature], tokenizer, MAX_LEN)
        
        # Structured features preprocessing
        numerical_features_for_scaling = [
            'sms_length', 'num_special_chars', 'num_digits', 'num_all_caps_words',
            'phishing_keywords_count', 'domain_age_days',
            'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
            'word_count', 'avg_word_length',
            'digits_to_length_ratio', 'special_chars_to_length_ratio', 'phishing_keywords_to_word_count_ratio'
        ]
        boolean_features_as_numerical = [
            'has_url', 'has_email', 'has_phone_number', 'is_ip_address_url',
            'url_and_ip', 'url_and_suspicious_tld', 'urgent_and_url'
        ]
        categorical_features_for_ohe = [
            'domain_encoded', 'tld_encoded', 'subdomain_encoded', 'url_subcategory_encoded',
            'domain_registrar_encoded', 'sender_encoded', 'sender_type_encoded',
            'brand_encoded', 'message_category_encoded', 'dataset_source_encoded'
        ]

        numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in X_train.columns]
        boolean_features_as_numerical = [col for col in boolean_features_as_numerical if col in X_train.columns]
        categorical_features_for_ohe = [col for col in categorical_features_for_ohe if col in X_train.columns]

        all_numerical_cols_for_scaling = numerical_features_for_scaling + boolean_features_as_numerical
        for col in all_numerical_cols_for_scaling:
            X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
            X_val.loc[:, col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
            X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
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
        if isinstance(X_train_structured_processed, (np.ndarray, pd.DataFrame)):
            X_train_structured_processed = tf.constant(X_train_structured_processed, dtype=tf.float32)
            X_val_structured_processed = tf.constant(X_val_structured_processed, dtype=tf.float32)
            X_test_structured_processed = tf.constant(X_test_structured_processed, dtype=tf.float32)
        else: # Assuming it's a sparse matrix from OneHotEncoder
            # Need to convert sparse matrix to dense numpy array first, then to tf.constant
            X_train_structured_processed = tf.constant(X_train_structured_processed.toarray(), dtype=tf.float32)
            X_val_structured_processed = tf.constant(X_val_structured_processed.toarray(), dtype=tf.float32)
            X_test_structured_processed = tf.constant(X_test_structured_processed.toarray(), dtype=tf.float32)

        y_train = tf.constant(y_train.values, dtype=tf.float32)
        y_val = tf.constant(y_val.values, dtype=tf.float32)
        y_test = tf.constant(y_test.values, dtype=tf.float32)

    except Exception as e:
        print(f"Error loading/preparing data for model: {e}")
        print("Please ensure 'unified_phishing_sms_dataset_processed.csv' exists and the previous script ran successfully.")
        print("Exiting as data is critical for model training.")
        exit()

    # --- Create ModelConfig instance and call the training method ---
    model_config = ModelConfig(
        transformer_model_name=TRANSFORMER_MODEL_NAME,
        max_len=MAX_LEN,
        learning_rate=2e-5,
        epochs=5,
        batch_size=32,
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
        y_test=y_test
    )

    trained_model, training_history = build_and_train_hybrid_model(model_config)

    # You can save the trained model after successful training
    trained_model.save('hybrid_phishing_detector_model.h5')
    print("\nModel saved as 'hybrid_phishing_detector_model.h5'")
    print("\n--- Model Training and Evaluation Complete ---")
    