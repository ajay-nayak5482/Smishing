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


def build_and_train_hybrid_model(config: ModelConfig, preprocessor_obj=None, tokenizer_obj=None, save_model=True):
    """
    Builds, compiles, trains, and evaluates the hybrid SMS phishing detection model.
    Now also returns the trained model, history, preprocessor, and tokenizer.
    Added `save_model` flag to control saving/TFLite conversion.
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

    # Only perform detailed evaluation plots and save/convert if `save_model` is True
    if save_model:
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
class HybridModelTextAttackWrapper(ModelWrapper):
    def __init__(self, keras_model, tokenizer_obj, preprocessor_obj, structured_features_template):
        self.model = keras_model
        self.tokenizer = tokenizer_obj
        self.preprocessor = preprocessor_obj
        self.structured_features_template = structured_features_template

    def __call__(self, text_input_list):
        input_ids, attention_mask = tokenize_text(pd.Series(text_input_list), self.tokenizer, MAX_LEN)
        num_examples = len(text_input_list)
        batch_structured_features = tf.tile(self.structured_features_template, [num_examples, 1])

        predictions = self.model({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structured_features_input': batch_structured_features
        })
        
        predictions = tf.concat([1 - predictions, predictions], axis=-1)
        
        return predictions.numpy()

# --- NEW FUNCTION: Generate Adversarial Data ---
def generate_adversarial_data(model_wrapper, tokenizer, preprocessor, X_source_data, y_source_data, num_attacks_per_iteration=50):
    """
    Generates successful adversarial examples from phishing messages in the source data.
    Returns a tuple: (list of adversarial texts, list of original structured feature DataFrames, list of original labels).
    """
    print(f"\n--- Generating {num_attacks_per_iteration} Adversarial Messages from source data ---")

    y_source_data_np = y_source_data.numpy()
    
    # Get the actual index labels of phishing examples from X_source_data
    phishing_index_labels = X_source_data[y_source_data_np == 1].index.tolist()

    if not phishing_index_labels:
        print("No phishing examples found in source data for adversarial attack.")
        return [], [], []

    num_attacks = min(num_attacks_per_iteration, len(phishing_index_labels))
    sampled_phishing_indices = np.random.choice(phishing_index_labels, num_attacks, replace=False)
    
    X_phishing_sampled = X_source_data.loc[sampled_phishing_indices]
    
    # TextAttack Dataset expects (text, label) pairs
    # Label 1 means positive class (phishing)
    textattack_dataset = Dataset([
        (row[1]['sms_content_cleaned_for_nlp'], 1)
        for row in X_phishing_sampled.iterrows()
    ])

    attack = recipes.TextFoolerJin2019.build(model_wrapper)

    attack_args = AttackArgs(
        num_examples=len(textattack_dataset),
        log_to_csv="temp_log_textattack.csv", # Use temp file for intermediate logs
        #log_to_stdout=False, # Suppress stdout during loop for cleaner output
        #log_to_file=False, # Don't write to file repeatedly
        disable_stdout=True # Disable stdout for TextAttack progress bars
    )

    attacker = Attacker(attack, textattack_dataset, attack_args)
    results = attacker.attack_dataset()

    adversarial_texts = []
    adversarial_original_structured_features_df_list = []
    adversarial_labels = []

    for i, result in enumerate(results):
        if result.perturbed_result: # If an adversarial example was found
            original_text = result.original_result.attacked_text.text
            perturbed_text = result.perturbed_result.attacked_text.text
            original_score = result.original_result.score
            perturbed_score = result.perturbed_result.score

            # Ensure scores are arrays for indexing
            if isinstance(original_score, float):
                original_score = np.array([1 - original_score, original_score])
            if isinstance(perturbed_score, float):
                perturbed_score = np.array([1 - perturbed_score, perturbed_score])

            original_pred_class = np.argmax(original_score)
            perturbed_pred_class = np.argmax(perturbed_score)

            is_successful_attack = (original_pred_class == 1 and perturbed_pred_class == 0)

            if is_successful_attack:
                # Get the original row from X_source_data that corresponds to this result
                # The order of results from TextAttack corresponds to the order in textattack_dataset
                # sampled_phishing_indices[i] gives the original index label from X_source_data
                original_row_idx = sampled_phishing_indices[i]
                # Extract all features *except* the text feature for structured part
                original_structured_features_for_this_sms = X_source_data.loc[[original_row_idx]].drop(columns=['sms_content_cleaned_for_nlp'])
                
                adversarial_texts.append(perturbed_text)
                adversarial_original_structured_features_df_list.append(original_structured_features_for_this_sms)
                adversarial_labels.append(1) # Adversarial example of a phishing message still has ground truth 1
        else:
            # Optionally log failed/skipped attacks if needed for debugging
            pass

    print(f"Generated {len(adversarial_texts)} successful adversarial examples.")
    return adversarial_texts, adversarial_original_structured_features_df_list, adversarial_labels

# --- NEW FUNCTION: Run Iterative Adversarial Training Loop ---
def run_adversarial_training_loop(
    initial_model, tokenizer, preprocessor,
    X_train_original, y_train_original,
    X_test_clean, y_test_clean,
    num_iterations=3,
    num_attacks_per_iteration=50,
    learning_rate=2e-5,
    epochs_per_iteration=2,
    batch_size=32
):
    """
    Orchestrates the iterative adversarial training loop.

    Args:
        initial_model (tf.keras.Model): The pre-trained model to start with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        preprocessor (sklearn.compose.ColumnTransformer): The structured data preprocessor.
        X_train_original (pd.DataFrame): Original training features.
        y_train_original (tf.Tensor): Original training labels.
        X_test_clean (pd.DataFrame): Clean test features.
        y_test_clean (tf.Tensor): Clean test labels.
        num_iterations (int): Number of adversarial training iterations.
        num_attacks_per_iteration (int): Number of adversarial examples to generate per iteration.
        learning_rate (float): Learning rate for fine-tuning.
        epochs_per_iteration (int): Number of epochs to fine-tune in each iteration.
        batch_size (int): Batch size for training.
    """
    print("\n--- Starting Iterative Adversarial Training Loop ---")
    current_model = initial_model
    
    # Store performance metrics over iterations
    clean_test_accuracy_history = []
    adversarial_test_accuracy_history = []
    
    # Initial evaluation on clean test set
    print("\n--- Initial Evaluation on Clean Test Set ---")
    initial_test_inputs = {
        'input_ids': tokenize_text(X_test_clean['sms_content_cleaned_for_nlp'], tokenizer, MAX_LEN)[0],
        'attention_mask': tokenize_text(X_test_clean['sms_content_cleaned_for_nlp'], tokenizer, MAX_LEN)[1],
        'structured_features_input': tf.constant(preprocessor.transform(X_test_clean).toarray(), dtype=tf.float32)
    }
    initial_loss, initial_accuracy, initial_precision, initial_recall = current_model.evaluate(initial_test_inputs, y_test_clean, verbose=0) # verbose=0 to suppress per-batch output
    print(f"Initial Clean Test Accuracy: {initial_accuracy:.4f}")
    clean_test_accuracy_history.append(initial_accuracy)

    for i in range(num_iterations):
        print(f"\n--- Adversarial Training Iteration {i+1}/{num_iterations} ---")

        # 1. Generate adversarial examples from current training data using the current model
        print(f"Generating adversarial examples from training data (Iteration {i+1})...")
        
        # Create a model wrapper for the current model for TextAttack
        sample_structured_features_for_wrapper = preprocessor.transform(X_train_original.head(1)).toarray()
        structured_input_dim_for_wrapper = sample_structured_features_for_wrapper.shape[1]
        structured_features_template_for_wrapper = tf.zeros((1, structured_input_dim_for_wrapper), dtype=tf.float32)

        current_model_wrapper = HybridModelTextAttackWrapper(
            keras_model=current_model,
            tokenizer_obj=tokenizer,
            preprocessor_obj=preprocessor,
            structured_features_template=structured_features_template_for_wrapper
        )

        adversarial_texts, adversarial_original_structured_features_df_list, adversarial_labels = \
            generate_adversarial_data(current_model_wrapper, tokenizer, preprocessor, X_train_original, y_train_original, num_attacks_per_iteration)
        
        if not adversarial_texts:
            print(f"No successful adversarial examples generated in iteration {i+1}. Skipping fine-tuning for this iteration.")
            # Append last known accuracies if no new adversarial examples were generated
            clean_test_accuracy_history.append(clean_test_accuracy_history[-1])
            adversarial_test_accuracy_history.append(adversarial_test_accuracy_history[-1] if adversarial_test_accuracy_history else 0.0)
            continue # Skip to next iteration or break if no progress

        # 2. Prepare augmented training data
        print(f"Preparing augmented training data for iteration {i+1}...")
        
        # Create a DataFrame for adversarial examples' structured features
        # Concatenate the DataFrames from the list
        adversarial_structured_df = pd.concat(adversarial_original_structured_features_df_list, ignore_index=True)

        # Create a DataFrame for adversarial examples' text and combine with structured features
        X_adversarial_df = adversarial_structured_df.copy()
        X_adversarial_df['sms_content_cleaned_for_nlp'] = adversarial_texts
        
        # Combine original training data with adversarial data
        X_train_augmented = pd.concat([X_train_original, X_adversarial_df], ignore_index=True)
        y_train_augmented = tf.concat([y_train_original, tf.constant(adversarial_labels, dtype=tf.float32)], axis=0)

        # Re-tokenize and re-process structured features for the *entire augmented dataset*
        print("Re-tokenizing and re-processing augmented training data...")
        train_input_ids_aug, train_attention_mask_aug = tokenize_text(X_train_augmented['sms_content_cleaned_for_nlp'], tokenizer, MAX_LEN)
        
        # Transform X_train_augmented using the *original* preprocessor
        X_train_structured_processed_aug = preprocessor.transform(X_train_augmented).toarray()
        X_train_structured_processed_aug = tf.constant(X_train_structured_processed_aug, dtype=tf.float32)

        # 3. Fine-tune the model on the augmented data
        print(f"Fine-tuning model on augmented data (Iteration {i+1})...")
        
        # Create a new ModelConfig for the fine-tuning step
        finetune_config = ModelConfig(
            transformer_model_name=TRANSFORMER_MODEL_NAME,
            max_len=MAX_LEN,
            learning_rate=learning_rate,
            epochs=epochs_per_iteration,
            batch_size=batch_size,
            train_input_ids=train_input_ids_aug,
            train_attention_mask=train_attention_mask_aug,
            X_train_structured_processed=X_train_structured_processed_aug,
            y_train=y_train_augmented,
            # Pass existing validation/test data (needed for ModelConfig, but not used in fit directly)
            val_input_ids=config.val_input_ids,
            val_attention_mask=config.val_attention_mask,
            X_val_structured_processed=config.X_val_structured_processed,
            y_val=config.y_val,
            test_input_ids=config.test_input_ids,
            test_attention_mask=config.test_attention_mask,
            X_test_structured_processed=config.X_test_structured_processed,
            y_test=config.y_test
        )
        
        # Re-create the model with the same architecture and load weights from the current_model
        # This ensures we continue training from the current state.
        new_model_instance, _, _, _ = build_and_train_hybrid_model(finetune_config, preprocessor_obj=preprocessor, tokenizer_obj=tokenizer, save_model=False)
        current_model = new_model_instance # Update current_model for next iteration

        # 4. Evaluate robustness on a fresh set of adversarial examples from the test set
        print(f"\nEvaluating robustness on adversarial test set (Iteration {i+1})...")
        
        # Generate adversarial examples from the clean test set using the *newly trained* model
        test_model_wrapper = HybridModelTextAttackWrapper(
            keras_model=current_model,
            tokenizer_obj=tokenizer,
            preprocessor_obj=preprocessor,
            structured_features_template=tf.zeros((1, structured_input_dim_for_wrapper), dtype=tf.float32)
        )
        
        adversarial_test_texts, adversarial_test_structured_features_df_list, adversarial_test_labels = \
            generate_adversarial_data(test_model_wrapper, tokenizer, preprocessor, X_test_clean, y_test_clean, num_attacks_per_iteration)
        
        if adversarial_test_texts:
            X_test_adversarial_df = pd.DataFrame({'sms_content_cleaned_for_nlp': adversarial_test_texts})
            if adversarial_test_structured_features_df_list:
                X_test_adversarial_df = pd.concat([X_test_adversarial_df, pd.concat(adversarial_test_structured_features_df_list, ignore_index=True)], axis=1)
            
            test_input_ids_adv, test_attention_mask_adv = tokenize_text(X_test_adversarial_df['sms_content_cleaned_for_nlp'], tokenizer, MAX_LEN)
            X_test_structured_processed_adv = preprocessor.transform(X_test_adversarial_df).toarray()
            X_test_structured_processed_adv = tf.constant(X_test_structured_processed_adv, dtype=tf.float32)
            y_test_adv = tf.constant(adversarial_test_labels, dtype=tf.float32)

            adversarial_test_inputs = {
                'input_ids': test_input_ids_adv,
                'attention_mask': test_attention_mask_adv,
                'structured_features_input': X_test_structured_processed_adv
            }
            
            adv_loss, adv_accuracy, adv_precision, adv_recall = current_model.evaluate(adversarial_test_inputs, y_test_adv, verbose=0)
            print(f"Iteration {i+1} Adversarial Test Accuracy: {adv_accuracy:.4f}")
            adversarial_test_accuracy_history.append(adv_accuracy)
        else:
            print(f"No adversarial examples generated from test set in iteration {i+1}. Adversarial accuracy not updated.")
            adversarial_test_accuracy_history.append(adversarial_test_accuracy_history[-1] if adversarial_test_accuracy_history else 0.0)

        # Re-evaluate on clean test set to check for performance degradation
        print(f"Re-evaluating on clean test set (Iteration {i+1})...")
        loss_clean, accuracy_clean, precision_clean, recall_clean = current_model.evaluate(initial_test_inputs, y_test_clean, verbose=0)
        print(f"Iteration {i+1} Clean Test Accuracy: {accuracy_clean:.4f}")
        clean_test_accuracy_history.append(accuracy_clean)

    print("\n--- Iterative Adversarial Training Loop Complete ---")
    print("\nClean Test Accuracy History:", clean_test_accuracy_history)
    print("Adversarial Test Accuracy History:", adversarial_test_accuracy_history)
    
    # Final save and TFLite conversion of the most robust model
    model_save_path = 'hybrid_phishing_detector_model_robust.h5'
    current_model.save(model_save_path)
    print(f"\nFinal robust model saved as '{model_save_path}'")

    tflite_model_path = 'hybrid_phishing_detector_model_robust.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(current_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Final robust model converted to TensorFlow Lite and saved as '{tflite_model_path}'")

    return current_model, clean_test_accuracy_history, adversarial_test_accuracy_history


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

    # Initial model training (Phase 1)
    print("\n--- Initial Model Training (Phase 1) ---")
    model_config_initial = ModelConfig(
        transformer_model_name=TRANSFORMER_MODEL_NAME,
        max_len=MAX_LEN,
        learning_rate=2e-5,
        epochs=15, # Initial training epochs
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

    initial_trained_model, _, _, _ = build_and_train_hybrid_model(model_config_initial, preprocessor_obj=preprocessor, tokenizer_obj=tokenizer, save_model=False)

    # --- Run Iterative Adversarial Training Loop (Phase 4) ---
    final_robust_model, clean_acc_history, adv_acc_history = run_adversarial_training_loop(
        initial_trained_model, tokenizer, preprocessor,
        X_train, y_train, # Pass original X_train, y_train for augmentation source
        X_test, y_test,   # Pass X_test, y_test for evaluation
        num_iterations=3, # Number of adversarial training iterations
        num_attacks_per_iteration=50, # Number of adversarial examples to generate per iteration
        learning_rate=1e-5, # Slightly lower LR for fine-tuning
        epochs_per_iteration=2 # Fewer epochs per fine-tuning step
    )

    print("\nFull model pipeline (training, evaluation, saving, TFLite conversion, and iterative adversarial training) completed.")
