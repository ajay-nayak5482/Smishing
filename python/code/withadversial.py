import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
from transformers import TFDistilBertModel, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings

# --- TextAttack Imports ---
import textattack
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
import textattack.attack_recipes as recipes # Import the recipes module

warnings.filterwarnings('ignore')

# --- Configuration ---
TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
DISTILBERT_HIDDEN_SIZE = 768

@dataclass
class ModelConfig:
    transformer_model_name: str
    max_len: int
    learning_rate: float
    epochs: int
    batch_size: int
    
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

def tokenize_text(texts, tokenizer, max_len):
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


def build_and_train_hybrid_model(config: ModelConfig, preprocessor_obj=None, tokenizer_obj=None):
    """
    Builds, compiles, trains, and evaluates the hybrid SMS phishing detection model.
    Now also returns the trained model, history, preprocessor, and tokenizer.
    """
    print("\n--- Defining Hybrid Model Architecture ---")

    input_ids = Input(shape=(config.max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(config.max_len,), dtype=tf.int32, name='attention_mask')

    distilbert_model_layer = TFDistilBertModel.from_pretrained(config.transformer_model_name, trainable=True, return_dict=False)

    def call_distilbert(inputs):
        input_ids_tensor, attention_mask_tensor = inputs
        return distilbert_model_layer(
            input_ids=input_ids_tensor, 
            attention_mask=attention_mask_tensor
        )
    
    transformer_outputs = Lambda(
        call_distilbert, 
        output_shape=[(config.max_len, DISTILBERT_HIDDEN_SIZE)],
        name='distilbert_wrapper'
    )([input_ids, attention_mask])
    
    text_features = transformer_outputs[0][:, 0, :]
    text_features = Dropout(0.2)(text_features)

    structured_input_dim = config.X_train_structured_processed.shape[1]
    structured_features_input = Input(shape=(structured_input_dim,), dtype=tf.float32, name='structured_features_input')

    structured_features_mlp = Dense(128, activation='relu')(structured_features_input)
    structured_features_mlp = Dropout(0.2)(structured_features_mlp)
    structured_features_mlp = Dense(64, activation='relu')(structured_features_mlp)
    structured_features_mlp = Dropout(0.2)(structured_features_mlp)

    combined_features = Concatenate()([text_features, structured_features_mlp])

    output = Dense(64, activation='relu')(combined_features)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_ids, attention_mask, structured_features_input], outputs=output)

    print("\n--- Compiling Model ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary()

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(config.y_train),
        y=config.y_train.numpy()
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"\nCalculated Class Weights: {class_weights_dict}")

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

    print("\n--- Performing Detailed Evaluation ---")
    y_pred_proba = model.predict(test_inputs).ravel()
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    cm = confusion_matrix(config.y_test, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Ham', 'Predicted Phishing'],
                yticklabels=['Actual Ham', 'Actual Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    fpr, tpr, thresholds = roc_curve(config.y_test, y_pred_proba)
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
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")

    model_save_path = 'hybrid_phishing_detector_model.h5'
    model.save(model_save_path)
    print(f"\nModel saved as '{model_save_path}'")

    tflite_model_path = 'hybrid_phishing_detector_model.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TensorFlow Lite and saved as '{tflite_model_path}'")

    # Return preprocessor and tokenizer along with the model and history
    return model, history, preprocessor_obj, tokenizer_obj

# --- Custom ModelWrapper for TextAttack ---
# This wrapper assumes structured features are provided as a fixed tensor for all attacked texts.
# In a real scenario, you'd want to generate/modify structured features for each adversarial text.
class HybridModelTextAttackWrapper(ModelWrapper):
    def __init__(self, keras_model, tokenizer_obj, preprocessor_obj, structured_features_template):
        self.model = keras_model
        self.tokenizer = tokenizer_obj
        self.preprocessor = preprocessor_obj
        # structured_features_template should be a tf.Tensor of shape (1, structured_input_dim)
        # It serves as the baseline for the structured features input for TextAttack.
        self.structured_features_template = structured_features_template

    def __call__(self, text_input_list):
        # Tokenize the text inputs provided by TextAttack
        input_ids, attention_mask = tokenize_text(pd.Series(text_input_list), self.tokenizer, MAX_LEN)

        # Replicate the structured_features_template for the current batch size
        num_examples = len(text_input_list)
        batch_structured_features = tf.tile(self.structured_features_template, [num_examples, 1])

        # Make prediction with the hybrid model
        predictions = self.model({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structured_features_input': batch_structured_features
        })
        
        # TextAttack expects raw logits or probabilities for each class.
        # Our model outputs a single sigmoid probability for the positive class (phishing).
        # Convert it to a 2-element array [prob_class_0, prob_class_1]
        predictions = tf.concat([1 - predictions, predictions], axis=-1)
        
        return predictions.numpy() # TextAttack expects a numpy array

def generate_and_evaluate_adversarial_examples(
    trained_model, tokenizer, preprocessor, X_data, y_data
):
    print("\n--- Phase 4: Adversarial Message Generation ---")

    # --- 1. Identify Phishing Examples for Attack ---
    y_data_np = y_data.numpy()
    
    # Get the actual index labels of phishing examples from X_data
    phishing_index_labels = X_data[y_data_np == 1].index.tolist()
    
    # Limit to a reasonable number for TextAttack, as it can be slow
    num_attacks = 2 # Adjust as needed for computation time
    if len(phishing_index_labels) > num_attacks:
        sampled_phishing_indices = np.random.choice(phishing_index_labels, num_attacks, replace=False)
    else:
        sampled_phishing_indices = phishing_index_labels
    
    X_phishing_sampled = X_data.loc[sampled_phishing_indices]
    y_phishing_sampled_series = pd.Series(y_data_np, index=X_data.index)
    y_phishing_sampled_np = y_phishing_sampled_series.loc[sampled_phishing_indices].values

    print(f"\nAttacking {len(sampled_phishing_indices)} phishing examples...")

    sample_structured_features = preprocessor.transform(X_phishing_sampled.head(1)).toarray()
    structured_input_dim = sample_structured_features.shape[1]
    
    structured_features_template = tf.zeros((1, structured_input_dim), dtype=tf.float32)

    # --- 2. Text-level Attacks (using TextAttack) ---
    print("\n--- Running Text-level Adversarial Attacks ---")

    attack_model_wrapper = HybridModelTextAttackWrapper(
        keras_model=trained_model,
        tokenizer_obj=tokenizer,
        preprocessor_obj=preprocessor,
        structured_features_template=structured_features_template
    )

    textattack_dataset = Dataset([
        (row[1]['sms_content_cleaned_for_nlp'], 1)
        for row in X_phishing_sampled.iterrows()
    ])

    # Corrected TextAttack recipe: Using TextFoolerJin2019
    attack = recipes.TextFoolerJin2019.build(attack_model_wrapper)
    # Alternatively, you can use other recipes like:
    # attack = recipes.PWWSRen2019.build(attack_model_wrapper)
    attack_args = AttackArgs(
        num_examples=len(textattack_dataset),
        log_to_csv="log_textattack.csv",
        #log_to_stdout=True,
        #log_to_file=True,
        disable_stdout=False
    )

    attacker = Attacker(attack, textattack_dataset, attack_args)
    results = attacker.attack_dataset()

    adversarial_text_examples = []
    attack_success_count_text = 0

    for result in results:
        original_text = result.original_result.attacked_text.text
        perturbed_text = result.perturbed_result.attacked_text.text if result.perturbed_result else None
        original_score = result.original_result.score
        perturbed_score = result.perturbed_result.score if result.perturbed_result else None
        
        # Ensure scores are arrays for indexing, assuming float is prob of positive class
        # Remove the 'if original_score' check as it's guaranteed to be a valid score
        if isinstance(original_score, float):
            original_score_array = np.array([1 - original_score, original_score])
        else:
            original_score_array = original_score # Already an array from HybridModelTextAttackWrapper
        
        if perturbed_score is not None:
            if isinstance(perturbed_score, float):
                perturbed_score_array = np.array([1 - perturbed_score, perturbed_score])
            else:
                perturbed_score_array = perturbed_score # Already an array
        else:
            perturbed_score_array = None # Keep as None if no perturbed result

        original_pred_class = np.argmax(original_score)
        perturbed_pred_class = np.argmax(perturbed_score) if perturbed_score is not None else original_pred_class

        is_successful_attack = (original_pred_class == 1 and perturbed_pred_class == 0)

        adversarial_text_examples.append({
            'original_sms': original_text,
            'adversarial_sms': perturbed_text,
             # Use the guaranteed array for original_prediction
            'original_prediction': f'Phishing (score: {original_score_array[1]:.4f})',
            # Use the potentially None array for adversarial_prediction, with a proper check
            'adversarial_prediction': f'Ham (score: {perturbed_score_array[1]:.4f})' if perturbed_score_array is not None else 'No change',
            'attack_type': 'Text-level',
            'attack_successful': is_successful_attack
        })
        if is_successful_attack:
            attack_success_count_text += 1
            print(f"  SUCCESS! Original: '{original_text}' -> Adversarial: '{perturbed_text}'")
        else:
            print(f"  FAILED. Original: '{original_text}' (pred: {original_pred_class}) -> Adversarial: '{perturbed_text}' (pred: {perturbed_pred_class})")

    if len(sampled_phishing_indices) > 0:
        text_attack_success_rate = attack_success_count_text / len(sampled_phishing_indices)
        print(f"\nText-level Attack Success Rate: {text_attack_success_rate:.2%}")
    else:
        text_attack_success_rate = 0.0
        print("\nNo phishing examples sampled for text attack.")


    # --- 3. Structured-feature Inspired Attacks ---
    print("\n--- Running Structured-feature Inspired Adversarial Attacks ---")
    print("This section currently provides a conceptual outline. Full implementation would involve manual or heuristic modification of features for *existing* phishing examples.")
    
    structured_attack_examples = []
    structured_attack_success_count = 0

    structured_attack_sample_df = X_phishing_sampled.sample(min(5, len(X_phishing_sampled)), random_state=42)

    for idx, original_row in structured_attack_sample_df.iterrows():
        original_sms = original_row['sms_content_cleaned_for_nlp']
        
        print(f"\nOriginal SMS for structured attack: '{original_sms}'")
        print(f"Original structured features (partial): Has URL: {original_row['has_url']}, Sender Type: {original_row['sender_type_encoded']}")
        
        original_structured_features = tf.constant(preprocessor.transform(pd.DataFrame([original_row])).toarray(), dtype=tf.float32)

        original_combined_inputs = {
            'input_ids': tokenize_text(pd.Series([original_sms]), tokenizer, MAX_LEN)[0],
            'attention_mask': tokenize_text(pd.Series([original_sms]), tokenizer, MAX_LEN)[1],
            'structured_features_input': original_structured_features
        }
        original_pred_proba = trained_model.predict(original_combined_inputs).ravel()[0]
        original_pred_class = (original_pred_proba > 0.5).astype(int)
        
    print("\nStructured-feature inspired attacks require careful feature inversion or generation, which is complex for this demo.")
    print("To proceed, you would define functions that take a raw SMS and its extracted structured features, and then modify specific feature values (e.g., changing a suspicious TLD to '.com', or a sender from 'unknown' to 'PayPal') and re-encode them.")
    print("Then, you would feed these new structured features (with the original SMS text) into the model to test the attack.")

    # --- 4. Evaluation of Adversarial Examples (Combined Summary) ---
    print("\n--- Summary of Adversarial Attack Results ---")
    print(f"Total phishing examples targeted for text attack: {len(sampled_phishing_indices)}")
    print(f"Text-level Attack Success Rate: {text_attack_success_rate:.2%}")

    # --- 5. Human Evaluation Guidance ---
    print("\n--- Guidance for Human Evaluation ---")
    print("For each successful adversarial example (especially text-level):")
    print("1.  **Semantic Preservation:** Read the original and adversarial SMS side-by-side.")
    print("    Does the adversarial SMS still convey the original intent of the phishing message?")
    print("    Is it still clearly a phishing attempt to a human?")
    print("2.  **Plausibility/Fluency:** Does the adversarial SMS look like a message a human would actually send or receive?")
    print("    Are there obvious typos or grammatical errors introduced by the attack that make it unrealistic?")
    ("    (TextAttack often introduces subtle changes, but sometimes they can be noticeable.)")
    print("\nManually review the 'adversarial_text_examples' list for successful attacks.")
    print("Focus on examples where 'attack_successful' is True.")
    print("This step is crucial to ensure the adversarial examples are realistic challenges to your model, not just random noise.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        unified_df = pd.read_csv('unified_phishing_sms_dataset_processed.csv')
        print("Loaded processed unified dataset for model building.")

        text_feature = 'sms_content_cleaned_for_nlp'
        target_feature = 'is_phishing'

        X = unified_df.drop(columns=[target_feature])
        y = unified_df[target_feature]

        # Use 80% for training+validation, 20% for test (from previous planning)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val) # 0.25 of 0.8 is 0.2

        # Initialize tokenizer (needed for both model building and adversarial generation)
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

        train_input_ids, train_attention_mask = tokenize_text(X_train[text_feature], tokenizer, MAX_LEN)
        val_input_ids, val_attention_mask = tokenize_text(X_val[text_feature], tokenizer, MAX_LEN)
        test_input_ids, test_attention_mask = tokenize_text(X_test[text_feature], tokenizer, MAX_LEN)
        
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

        # Ensure columns exist before including them
        numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in X_train.columns]
        boolean_features_as_numerical = [col for col in boolean_features_as_numerical if col in X_train.columns]
        categorical_features_for_ohe = [col for col in categorical_features_for_ohe if col in X_train.columns]

        all_numerical_cols_for_scaling = numerical_features_for_scaling + boolean_features_as_numerical
        for col in all_numerical_cols_for_scaling:
            X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
            X_val.loc[:, col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
            X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        # Initialize preprocessor (needed for both model building and adversarial generation)
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

        if isinstance(X_train_structured_processed, (np.ndarray, pd.DataFrame)):
            X_train_structured_processed = tf.constant(X_train_structured_processed, dtype=tf.float32)
            X_val_structured_processed = tf.constant(X_val_structured_processed, dtype=tf.float32)
            X_test_structured_processed = tf.constant(X_test_structured_processed, dtype=tf.float32)
        else:
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

    model_config = ModelConfig(
        transformer_model_name=TRANSFORMER_MODEL_NAME,
        max_len=MAX_LEN,
        learning_rate=2e-5,
        epochs=1,
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

    trained_model, training_history, final_preprocessor, final_tokenizer = build_and_train_hybrid_model(model_config, preprocessor_obj=preprocessor, tokenizer_obj=tokenizer)

    # --- Run Adversarial Generation Phase ---
    generate_and_evaluate_adversarial_examples(
        trained_model, final_tokenizer, final_preprocessor, X_test, y_test
    )

    print("\nFull model pipeline (training, evaluation, saving, TFLite conversion, and initial adversarial generation) completed.")
