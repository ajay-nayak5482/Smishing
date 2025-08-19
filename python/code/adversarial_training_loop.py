import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os

# Import modules
from config import (
    MAX_LEN, TRANSFORMER_MODEL_NAME, ADVERSARIAL_LEARNING_RATE, 
    EPOCHS_PER_ADVERSARIAL_ITERATION, BATCH_SIZE, NUM_ADVERSARIAL_ITERATIONS,
    NUM_ATTACKS_PER_ITERATION, TEXT_FEATURE,
    ROBUST_MODEL_PREFIX, FINAL_ROBUST_MODEL_PATH, FINAL_TFLITE_MODEL_PATH,
    TRAINING_HISTORY_PATH, PLOTS_DIR, ADVERSARIAL_ACC_HISTORY_PLOT_PATH,
    ANDROID_VOCAB_PATH, ANDROID_SCALER_PARAMS_PATH,
    ANDROID_ENCODER_PARAMS_PATH, ANDROID_FEATURE_ORDER_PATH, EXPORTED_MODEL_DIR,
    INITIAL_TRAINING_HISOTRY
)
from model_architecture import ModelConfig, build_and_train_hybrid_model
from adversarial_utils import HybridModelTextAttackWrapper, generate_adversarial_data
from data_preparation import tokenize_text, plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_adversarial_accuracy_history
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Needed for exporting preprocessor details
import pickle # For saving preprocessor


def run_adversarial_training_loop(
    initial_model, tokenizer, preprocessor,
    X_train_original_df, y_train_original_tf,
    X_test_clean_df, y_test_clean_tf
):
    """
    Orchestrates the iterative adversarial training loop to enhance model robustness.

    Args:
        initial_model (tf.keras.Model): The pre-trained model to start with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        preprocessor (sklearn.compose.ColumnTransformer): The structured data preprocessor.
        X_train_original_df (pd.DataFrame): Original training features DataFrame.
        y_train_original_tf (tf.Tensor): Original training labels TensorFlow Tensor.
        X_test_clean_df (pd.DataFrame): Clean test features DataFrame.
        y_test_clean_tf (tf.Tensor): Clean test labels TensorFlow Tensor.

    Returns:
        tuple: (tf.keras.Model, list, list) The final robust model,
               history of clean test accuracies, history of adversarial test accuracies.
    """
    print("\n--- Starting Iterative Adversarial Training Loop ---")
    current_model = initial_model
    
    # Load history if it exists, otherwise initialize
    if os.path.exists(TRAINING_HISTORY_PATH):
        with open(TRAINING_HISTORY_PATH, 'r') as f:
            history_data = json.load(f)
            clean_test_accuracy_history = history_data.get('clean_test_accuracy_history', [])
            adversarial_test_accuracy_history = history_data.get('adversarial_test_accuracy_history', [])
            # Adjust start_iteration based on how many full iterations were completed
            start_iteration = len(adversarial_test_accuracy_history) # Number of completed iterations
            print(f"Resuming adversarial training from iteration {start_iteration + 1}...")
    else:
        clean_test_accuracy_history = []
        adversarial_test_accuracy_history = []
        start_iteration = 0
    
    # Initial evaluation on clean test set if not already recorded (only if starting fresh)
    # if start_iteration == 0:
    print("\n--- Initial Evaluation on Clean Test Set ---")
    # Process clean test data for evaluation inputs
    test_input_ids_clean, test_attention_mask_clean = tokenize_text(X_test_clean_df[TEXT_FEATURE], tokenizer, MAX_LEN)
    X_test_structured_processed_clean = preprocessor.transform(X_test_clean_df).toarray()
    X_test_structured_processed_clean = tf.constant(X_test_structured_processed_clean, dtype=tf.float32)

    initial_test_inputs = {
        'input_ids': test_input_ids_clean,
        'attention_mask': test_attention_mask_clean,
        'structured_features_input': X_test_structured_processed_clean
    }
    initial_loss, initial_accuracy, initial_precision, initial_recall = current_model.evaluate(initial_test_inputs, y_test_clean_tf, verbose=0)
    print(f"Initial Clean Test Accuracy: {initial_accuracy:.4f}")
    clean_test_accuracy_history.append(initial_accuracy)
    adversarial_test_accuracy_history.append(0.0) # Placeholder for initial adversarial accuracy

    for i in range(start_iteration, NUM_ADVERSARIAL_ITERATIONS):
        print(f"\n--- Adversarial Training Iteration {i+1}/{NUM_ADVERSARIAL_ITERATIONS} ---")

        # 1. Generate adversarial examples from current training data using the current model
        print(f"Generating adversarial examples from training data (Iteration {i+1})...")
        
        # Create a model wrapper for the current model for TextAttack
        # Need a dummy structured features template for the wrapper, as TextAttack only perturbs text.
        if not X_train_original_df.empty:
            # Ensure the structured features for the wrapper template are correctly extracted from the DataFrame
            sample_structured_df_for_wrapper = X_train_original_df.head(1).drop(columns=[TEXT_FEATURE], errors='ignore')
            sample_structured_features_for_wrapper = preprocessor.transform(sample_structured_df_for_wrapper).toarray()
            structured_input_dim_for_wrapper = sample_structured_features_for_wrapper.shape[1]
            structured_features_template_for_wrapper = tf.zeros((1, structured_input_dim_for_wrapper), dtype=tf.float32)
        else:
            print("Warning: X_train_original_df is empty. Cannot determine structured input dimension for wrapper. Using placeholder.")
            # Fallback for structured_features_template_for_wrapper if X_train_original_df is empty
            structured_features_template_for_wrapper = tf.zeros((1, 10), dtype=tf.float32) # Placeholder, needs to be a reasonable size


        current_model_wrapper = HybridModelTextAttackWrapper(
            keras_model=current_model,
            tokenizer_obj=tokenizer,
            preprocessor_obj=preprocessor,
            structured_features_template=structured_features_template_for_wrapper
        )

        adversarial_texts, adversarial_original_structured_features_df_list, adversarial_labels = \
            generate_adversarial_data(current_model_wrapper, tokenizer, preprocessor, X_train_original_df, y_train_original_tf, i+1, NUM_ATTACKS_PER_ITERATION)
        
        if not adversarial_texts:
            print(f"No successful adversarial examples generated in iteration {i+1}. Skipping fine-tuning for this iteration.")
            clean_test_accuracy_history.append(clean_test_accuracy_history[-1])
            adversarial_test_accuracy_history.append(adversarial_test_accuracy_history[-1])
            # Save history after each iteration, even if skipped
            with open(TRAINING_HISTORY_PATH, 'w') as f:
                json.dump({'clean_test_accuracy_history': clean_test_accuracy_history, 
                           'adversarial_test_accuracy_history': adversarial_test_accuracy_history}, f)
            continue # Skip to next iteration

        # 2. Prepare augmented training data
        print(f"Preparing augmented training data for iteration {i+1}...")
        
        # Create a DataFrame for adversarial examples' structured features
        adversarial_structured_df = pd.concat(adversarial_original_structured_features_df_list, ignore_index=True)

        # Create a DataFrame for adversarial examples' text and combine with structured features
        X_adversarial_df = adversarial_structured_df.copy()
        X_adversarial_df[TEXT_FEATURE] = adversarial_texts
        
        # Combine original training data with adversarial data
        X_train_augmented = pd.concat([X_train_original_df, X_adversarial_df], ignore_index=True)
        y_train_augmented = tf.concat([y_train_original_tf, tf.constant(adversarial_labels, dtype=tf.float32)], axis=0)

        # Re-tokenize and re-process structured features for the *entire augmented dataset*
        print("Re-tokenizing and re-processing augmented training data...")
        train_input_ids_aug, train_attention_mask_aug = tokenize_text(X_train_augmented[TEXT_FEATURE], tokenizer, MAX_LEN)
        
        # Transform X_train_augmented using the *original* preprocessor
        X_train_structured_processed_aug = preprocessor.transform(X_train_augmented).toarray()
        X_train_structured_processed_aug = tf.constant(X_train_structured_processed_aug, dtype=tf.float32)

        # 3. Fine-tune the model on the augmented data
        print(f"Fine-tuning model on augmented data (Iteration {i+1})...")
        
        # Create a new ModelConfig for the fine-tuning step
        finetune_config = ModelConfig(
            transformer_model_name=TRANSFORMER_MODEL_NAME,
            max_len=MAX_LEN,
            learning_rate=ADVERSARIAL_LEARNING_RATE,
            epochs=INITIAL_TRAINING_HISOTRY.epochs_already_trained + EPOCHS_PER_ADVERSARIAL_ITERATION,
            epochs_already=INITIAL_TRAINING_HISOTRY.epochs_already_trained,  # Reset epochs for fine-tuning
            batch_size=BATCH_SIZE,            
            train_input_ids=train_input_ids_aug,
            train_attention_mask=train_attention_mask_aug,
            X_train_structured_processed=X_train_structured_processed_aug,
            y_train=y_train_augmented,
            # Pass existing validation/test data for ModelConfig (not used in fit directly)
            val_input_ids=initial_test_inputs['input_ids'], # Using test data as val for ModelConfig
            val_attention_mask=initial_test_inputs['attention_mask'],
            X_val_structured_processed=initial_test_inputs['structured_features_input'],
            y_val=y_test_clean_tf,
            test_input_ids=initial_test_inputs['input_ids'], # Using test data as test for ModelConfig
            test_attention_mask=initial_test_inputs['attention_mask'],
            X_test_structured_processed=initial_test_inputs['structured_features_input'],
            y_test=y_test_clean_tf
        )
        
        # Define a checkpoint path for this iteration's model
        iteration_model_path = f"{ROBUST_MODEL_PREFIX}{i+1}.keras"
        
        # Build and train, saving the best model for this iteration
        new_model_instance, _ = build_and_train_hybrid_model(finetune_config, preprocessor_obj=preprocessor, tokenizer_obj=tokenizer, save_model=True, model_save_path=iteration_model_path, trained_model=current_model)
        current_model = new_model_instance # Update current_model for next iteration
        INITIAL_TRAINING_HISOTRY.epochs_already_trained += EPOCHS_PER_ADVERSARIAL_ITERATION # Update epochs already trained

        # 4. Evaluate robustness on a fresh set of adversarial examples from the test set
        print(f"Evaluating robustness on adversarial test set (Iteration {i+1})...")
        
        # Generate adversarial examples from the clean test set using the *newly trained* model
        test_model_wrapper = HybridModelTextAttackWrapper(
            keras_model=current_model,
            tokenizer_obj=tokenizer,
            preprocessor_obj=preprocessor,
            structured_features_template=tf.zeros((1, structured_input_dim_for_wrapper), dtype=tf.float32)
        )
        
        adversarial_test_texts, adversarial_test_structured_features_df_list, adversarial_test_labels = \
            generate_adversarial_data(test_model_wrapper, tokenizer, preprocessor, X_test_clean_df, y_test_clean_tf, f"test_iter_{i+1}", NUM_ATTACKS_PER_ITERATION)
        
        if adversarial_test_texts:
            X_test_adversarial_structured_df = pd.concat(adversarial_test_structured_features_df_list, ignore_index=True)

            X_test_adversarial_df = X_test_adversarial_structured_df.copy()
            X_test_adversarial_df[TEXT_FEATURE] = adversarial_test_texts
            
            test_input_ids_adv, test_attention_mask_adv = tokenize_text(X_test_adversarial_df[TEXT_FEATURE], tokenizer, MAX_LEN)
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
            adversarial_test_accuracy_history.append(adversarial_test_accuracy_history[-1])

        # Re-evaluate on clean test set to check for performance degradation
        print(f"Re-evaluating on clean test set (Iteration {i+1})...")
        loss_clean, accuracy_clean, precision_clean, recall_clean = current_model.evaluate(initial_test_inputs, y_test_clean_tf, verbose=0)
        print(f"Iteration {i+1} Clean Test Accuracy: {accuracy_clean:.4f}")
        clean_test_accuracy_history.append(accuracy_clean)

        # Save history after each iteration
        with open(TRAINING_HISTORY_PATH, 'w') as f:
            json.dump({'clean_test_accuracy_history': clean_test_accuracy_history, 
                       'adversarial_test_accuracy_history': adversarial_test_accuracy_history}, f)

    print("\n--- Iterative Adversarial Training Loop Complete ---")
    print("\nClean Test Accuracy History:", clean_test_accuracy_history)
    print("Adversarial Test Accuracy History:", adversarial_test_accuracy_history)
    
    # Final save and TFLite conversion of the most robust model
    current_model.save(FINAL_ROBUST_MODEL_PATH)
    print(f"\nFinal robust model saved as '{FINAL_ROBUST_MODEL_PATH}'")

    converter = tf.lite.TFLiteConverter.from_keras_model(current_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
        ,tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    os.makedirs(EXPORTED_MODEL_DIR, exist_ok=True) # Ensure export directory exists
    with open(FINAL_TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Final robust model converted to TensorFlow Lite and saved as '{FINAL_TFLITE_MODEL_PATH}'")

    # Export preprocessing assets for Android
    print("\n--- Exporting Preprocessing Assets for Android ---")
    # 1. Export Tokenizer Vocabulary
    # vocab_dir = os.path.dirname(ANDROID_VOCAB_PATH)
    # os.makedirs(vocab_dir, exist_ok=True)
    tokenizer.save_vocabulary(EXPORTED_MODEL_DIR)
    print(f"Tokenizer vocabulary saved to {ANDROID_VOCAB_PATH}")

    # 2. Export StandardScaler parameters (mean and scale)
    scaler_params = {
        'mean': preprocessor.named_transformers_['num'].mean_.tolist(),
        'scale': preprocessor.named_transformers_['num'].scale_.tolist(),
        'feature_names_in': preprocessor.named_transformers_['num'].feature_names_in_.tolist()
    }
    with open(ANDROID_SCALER_PARAMS_PATH, 'w') as f:
        json.dump(scaler_params, f)
    print(f"StandardScaler parameters saved to {ANDROID_SCALER_PARAMS_PATH}")

    # 3. Export OneHotEncoder categories
    encoder_params = {
        'categories': [cat.tolist() for cat in preprocessor.named_transformers_['cat'].categories_],
        'feature_names_out': preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
    }
    with open(ANDROID_ENCODER_PARAMS_PATH, 'w') as f:
        json.dump(encoder_params, f)
    print(f"OneHotEncoder categories saved to {ANDROID_ENCODER_PARAMS_PATH}")

    # 4. Export the final structured feature order
    final_structured_feature_order = preprocessor.get_feature_names_out().tolist()
    with open(ANDROID_FEATURE_ORDER_PATH, 'w') as f:
        json.dump(final_structured_feature_order, f)
    print(f"Structured feature order saved to {ANDROID_FEATURE_ORDER_PATH}")
    
    return current_model, clean_test_accuracy_history, adversarial_test_accuracy_history