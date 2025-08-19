import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json

# Import all necessary modules
from config import (
    INITIAL_LEARNING_RATE, INITIAL_EPOCHS, BATCH_SIZE,
    NUM_ADVERSARIAL_ITERATIONS, NUM_ATTACKS_PER_ITERATION,
    ADVERSARIAL_LEARNING_RATE, EPOCHS_PER_ADVERSARIAL_ITERATION,
    UNIFIED_DATA_PATH, INITIAL_MODEL_PATH, FINAL_ROBUST_MODEL_PATH,
    TRAINING_HISTORY_PATH, PLOTS_DIR, MAX_LEN, # Import MAX_LEN from config
    NAME_DATASET1, NAME_DATASET2, NAME_DATASET3, PHISHING_KEYWORDS, # Raw data config
    INITIAL_TRAINING_HISOTRY
)

from data_preparation import prepare_data_for_model_input, integrate_and_preprocess_datasets, plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_adversarial_accuracy_history
from model_architecture import build_and_train_hybrid_model, ModelConfig, DistilBertKerasModel # Import DistilBertKerasModel
from adversarial_training_loop import run_adversarial_training_loop
from transformers import TFDistilBertModel # Needed for custom_objects in load_model


if __name__ == "__main__":
    print("\n--- Checking for GPU and configuring CPU ---")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {gpus[0].name}")
            # Optional: Set memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print("No GPU detected. Using CPU for training.")
            # Explicitly set CPU for parallel operations
            tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
            tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
            print(f"Configured to use {os.cpu_count()} CPU cores.")
    except Exception as e:
        print(f"Could not configure hardware due to error: {e}")
    # Set the working directory to the script's directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    try:
        # 1. Data Preparation, Cleaning, Feature Engineering (with checkpoint)
        # Check if unified data already exists
        if os.path.exists(UNIFIED_DATA_PATH):
            print(f"\n--- Loading unified and processed data from checkpoint: {UNIFIED_DATA_PATH} ---")
            unified_df = pd.read_csv(UNIFIED_DATA_PATH)
            print(f"Loaded {len(unified_df)} rows from checkpoint.")
        else:
            print("\n--- Starting Data Integration and Preprocessing (Phase 1) ---")
            # Define raw data paths relative to the project root
            raw_data_dir = os.path.join('..', 'data', 'raw')
            path_dataset1 = os.path.join(raw_data_dir, NAME_DATASET1)
            path_dataset2 = os.path.join(raw_data_dir, NAME_DATASET2)
            path_dataset3 = os.path.join(raw_data_dir, NAME_DATASET3)
            
            unified_df = integrate_and_preprocess_datasets(
                path_dataset1, path_dataset2, path_dataset3, PHISHING_KEYWORDS,
                output_path=UNIFIED_DATA_PATH
            )
            if unified_df is None:
                print("Data integration failed. Exiting.")
                exit()

        # 2. Prepare Data for Model Input (with checkpoint logic if needed, but usually fast enough to re-run)
        processed_data = prepare_data_for_model_input(unified_df)
        initial_trained_model = None  # Initialize model variable
        # 3. Initial Model Training (Phase 1)
        # Check if initial model already exists to skip retraining
        if os.path.exists(INITIAL_MODEL_PATH):
            print(f"\n--- in main script Loading initial model from checkpoint: {INITIAL_MODEL_PATH} ---")
            initial_trained_model = tf.keras.models.load_model(
                INITIAL_MODEL_PATH, 
                custom_objects={'DistilBertKerasModel': DistilBertKerasModel} # Add custom class
            )
            print("Initial model loaded successfully.")

            # Load initial training history if available
        if INITIAL_TRAINING_HISOTRY.remaining_epochs() == 0:
            print("No need to train more epochs")
            # Re-evaluate to show its pe
            # rformance
            initial_test_inputs = {
                'input_ids': processed_data.test_input_ids,
                'attention_mask': processed_data.test_attention_mask,
                'structured_features_input': processed_data.X_test_structured_processed
            }

            # print(f"{processed_data.test_input_ids.shape=}, {processed_data.test_attention_mask.shape=}, {processed_data.X_test_structured_processed.shape= }")
            # loss, accuracy, precision, recall = initial_trained_model.evaluate(initial_test_inputs, processed_data.y_test)
            # print(f"Loaded Model Test Loss: {loss:.4f}")
            # print(f"Loaded Model Test Accuracy: {accuracy:.4f}")
            # print(f"Loaded Model Test Precision: {precision:.4f}")
            # print(f"Loaded Model Test Recall: {recall:.4f}")

            # # Plot initial model's performance if not already done
            # y_pred_proba = initial_trained_model.predict(initial_test_inputs).ravel()
            # y_pred_class = (y_pred_proba > 0.5).astype(int)
            # plot_confusion_matrix(processed_data.y_test.numpy(), y_pred_class, save_path=os.path.join(PLOTS_DIR, 'initial_confusion_matrix.png'))
            # plot_roc_curve(processed_data.y_test.numpy(), y_pred_proba, save_path=os.path.join(PLOTS_DIR, 'initial_roc_curve.png'))

        else:
            print("\n--- Initial Model Training (Phase 1) ---")
            model_config_initial = ModelConfig(
                transformer_model_name=processed_data.tokenizer.name_or_path,
                max_len=MAX_LEN, # Use MAX_LEN from config.py directly
                learning_rate=INITIAL_LEARNING_RATE,
                epochs=INITIAL_EPOCHS,                    
                epochs_already=INITIAL_TRAINING_HISOTRY.epochs_already_trained,
                batch_size=BATCH_SIZE,
                train_input_ids=processed_data.train_input_ids,
                train_attention_mask=processed_data.train_attention_mask,
                X_train_structured_processed=processed_data.X_train_structured_processed,
                y_train=processed_data.y_train,
                val_input_ids=processed_data.val_input_ids,
                val_attention_mask=processed_data.val_attention_mask,
                X_val_structured_processed=processed_data.X_val_structured_processed,
                y_val=processed_data.y_val,
                test_input_ids=processed_data.test_input_ids,
                test_attention_mask=processed_data.test_attention_mask,
                X_test_structured_processed=processed_data.X_test_structured_processed,
                y_test=processed_data.y_test
            )

            initial_trained_model, initial_history = build_and_train_hybrid_model(
                model_config_initial, 
                preprocessor_obj=processed_data.preprocessor, 
                tokenizer_obj=processed_data.tokenizer, 
                save_model=True, # Save initial model and plots
                model_save_path=INITIAL_MODEL_PATH, # Specify save path
                trained_model=initial_trained_model
            )
            plot_training_history(initial_history, save_path=os.path.join(PLOTS_DIR, 'initial_training_history.png'))
            INITIAL_TRAINING_HISOTRY.epochs_already_trained = INITIAL_EPOCHS
            INITIAL_TRAINING_HISOTRY.save_to_json()  # Save updated training history


        # 4. Iterative Adversarial Training Loop (Phase 4)
        print("\n--- Starting Iterative Adversarial Training Loop (Phase 4) ---")
        final_robust_model, clean_acc_history, adv_acc_history = run_adversarial_training_loop(
            initial_trained_model, 
            processed_data.tokenizer, 
            processed_data.preprocessor,
            X_train_original_df=processed_data.X_train_original_df, 
            y_train_original_tf=processed_data.y_train,
            X_test_clean_df=processed_data.X_test_original_df,     
            y_test_clean_tf=processed_data.y_test
        )
        plot_adversarial_accuracy_history(clean_acc_history, adv_acc_history, save_path=os.path.join(PLOTS_DIR, 'adversarial_training_history.png'))

        print("\nFull model pipeline (training, evaluation, saving, TFLite conversion, and iterative adversarial training) completed.")
        print("Final Clean Test Accuracy History:", clean_acc_history)
        print("Final Adversarial Test Accuracy History:", adv_acc_history)

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()
