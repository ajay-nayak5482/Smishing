import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import TFDistilBertModel, DistilBertConfig
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataclasses import dataclass
import re # Import regex for name normalization
import os

# Import configurations
from config import TRANSFORMER_MODEL_NAME, MAX_LEN, DISTILBERT_HIDDEN_SIZE, PLOTS_DIR

# Import tokenize_text and plotting functions from data_preparation
from data_preparation import tokenize_text, plot_confusion_matrix, plot_roc_curve, plot_training_history

@dataclass
class ModelConfig:
    """
    Configuration parameters for building and training the hybrid model.
    """
    transformer_model_name: str
    max_len: int
    learning_rate: float
    epochs: int    
    epochs_already: int
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

    # Test data (needed for evaluation, but not directly for training in fit)
    test_input_ids: tf.Tensor
    test_attention_mask: tf.Tensor
    X_test_structured_processed: tf.Tensor
    y_test: tf.Tensor

@tf.keras.utils.register_keras_serializable()
class DistilBertKerasModel(tf.keras.Model):
    """
    A custom Keras Model subclass to wrap TFDistilBertModel.
    This handles the instantiation and weight loading/slicing internally,
    allowing it to be used seamlessly in the Keras functional API with custom MAX_LEN.
    """
    def __init__(self, model_name, max_len, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_len = max_len

        # 1. Load the base configuration
        distilbert_config = DistilBertConfig.from_pretrained(self.model_name)
        # 2. Modify the max_position_embeddings in the config to match our desired MAX_LEN
        distilbert_config.max_position_embeddings = self.max_len

        # 3. Instantiate the TFDistilBertModel *with this custom configuration*.
        # This builds the model's graph with the correct input shape (self.max_len).
        self.distilbert = TFDistilBertModel(distilbert_config)

        # Call the model with dummy inputs to build its weights before setting them
        # This is necessary for self.distilbert.weights to be populated
        dummy_input_ids = tf.zeros((1, self.max_len), dtype=tf.int32)
        dummy_attention_mask = tf.zeros((1, self.max_len), dtype=tf.int32)
        _ = self.distilbert(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

        # 4. Load the pre-trained weights manually, adapting positional embeddings.
        temp_model_for_weights = TFDistilBertModel.from_pretrained(self.model_name, trainable=False)
        
        pretrained_weight_values = temp_model_for_weights.get_weights()
        pretrained_weight_variables = temp_model_for_weights.weights

        new_model_weight_variables = self.distilbert.weights
        new_model_weight_values_init = self.distilbert.get_weights() 

        loaded_weights = []
        pretrained_name_to_value = {}
        for i, w_var in enumerate(pretrained_weight_variables):
            cleaned_name = re.sub(r':0$', '', w_var.name)
            cleaned_name = re.sub(r'tf_distil_bert_model(_\d+)?/', '', cleaned_name)
            pretrained_name_to_value[cleaned_name] = pretrained_weight_values[i]

        print("\n--- from model_architecture.py Attempting to load pre-trained weights by name and adapt for MAX_LEN ---")

        for new_weight_var in new_model_weight_variables:
            cleaned_new_name = re.sub(r':0$', '', new_weight_var.name)
            cleaned_new_name = re.sub(r'tf_distil_bert_model(_\d+)?/', '', cleaned_new_name)

            if cleaned_new_name in pretrained_name_to_value:
                old_weight_value = pretrained_name_to_value[cleaned_new_name]
                
                if "embeddings/position_embeddings/embeddings" in cleaned_new_name:
                    if old_weight_value.shape[0] > self.max_len:
                        loaded_weights.append(old_weight_value[:self.max_len, :])
                        print(f"Sliced positional embeddings from {old_weight_value.shape} to ({self.max_len}, {old_weight_value.shape[1]}) for {new_weight_var.name}")
                    else:
                        loaded_weights.append(old_weight_value)
                else:
                    if old_weight_value.shape == new_weight_var.shape:
                        loaded_weights.append(old_weight_value)
                    else:
                        print(f"Shape mismatch for weight {new_weight_var.name}: pretrained {old_weight_value.shape} vs new {new_weight_var.shape}. Using new model's initialized weight.")
                        loaded_weights.append(new_model_weight_values_init[new_model_weight_variables.index(new_weight_var)])
            else:
                print(f"Warning: No matching pre-trained weight found for {new_weight_var.name}. Using new model's initialized weight.")
                loaded_weights.append(new_model_weight_values_init[new_model_weight_variables.index(new_weight_var)])
        
        self.distilbert.set_weights(loaded_weights)
        self.distilbert.trainable = True # Ensure it's trainable
        temp_model_for_weights = None # Free up memory

    def call(self, inputs):
        input_ids, attention_mask = inputs
        # Directly call the internal TFDistilBertModel
        return self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def get_config(self): # Added get_config for serialization
        config = super().get_config()
        # --- MODIFICATION START ---
        # Ensure model_name and max_len are included in the config for serialization
        config.update({
            "model_name": self.model_name,
            "max_len": self.max_len,
        })
        # --- MODIFICATION END ---
        return config

    @classmethod # Added from_config for deserialization
    def from_config(cls, config):
        return cls(config["model_name"], config["max_len"])


def build_and_train_hybrid_model(config: ModelConfig, preprocessor_obj=None, tokenizer_obj=None, save_model=True, model_save_path=None, trained_model=None):
    """
    Builds, compiles, trains, and evaluates the hybrid SMS phishing detection model.
    Added `save_model` flag to control saving/TFLite conversion.
    `model_save_path` is now an explicit argument for checkpointing.
    """
    print("\n--- Defining Hybrid Model Architecture ---")

    # Text Branch Input
    input_ids = Input(shape=(config.max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(config.max_len,), dtype=tf.int32, name='attention_mask')

    # Use our custom wrapper model for DistilBERT
    distilbert_keras_model = DistilBertKerasModel(config.transformer_model_name, config.max_len, name='distilbert_text_encoder')
    transformer_outputs = distilbert_keras_model((input_ids, attention_mask))
    
    # Extract the [CLS] token's embedding (first token, first dimension)
    text_features = transformer_outputs[:, 0, :]

    text_features = Dropout(0.2)(text_features)

    # Structured Features Branch Input
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
    output = Dense(1, activation='sigmoid')(output)

    # Create the Hybrid Model
    model = Model(inputs=[input_ids, attention_mask, structured_features_input], outputs=output)
    if trained_model is not None:
        print("\n--- Loading Weights from Provided Trained Model ---")
        # If a trained model is provided, use its inputs and outputs        
        model.set_weights(trained_model.get_weights())    
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

    callbacks = []
    if save_model and model_save_path:
        # Save only the best model based on validation accuracy
        checkpoint_callback = ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',            
            verbose=1
        )
        callbacks.append(checkpoint_callback)


    history = model.fit(
        train_inputs,
        config.y_train,
        validation_data=(val_inputs, config.y_val),
        epochs=config.epochs,
        initial_epoch=config.epochs_already,
        batch_size=config.batch_size,
        class_weight=class_weights_dict,
        callbacks=callbacks # Add the checkpoint callback here
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

        # Plot and save Confusion Matrix
        plot_confusion_matrix(config.y_test.numpy(), y_pred_class, save_path=os.path.join(PLOTS_DIR, 'confusion_matrix.png'))

        # Plot and save ROC Curve
        roc_auc = plot_roc_curve(config.y_test.numpy(), y_pred_proba, save_path=os.path.join(PLOTS_DIR, 'roc_curve.png'))
        print(f"ROC AUC: {roc_auc:.4f}")

        # Plot and save training history (accuracy and loss)
        plot_training_history(history, save_path=os.path.join(PLOTS_DIR, 'training_history.png'))

    return model, history
