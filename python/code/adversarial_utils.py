# adversarial_utils.py

import tensorflow as tf
import numpy as np
import pandas as pd
import textattack
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
import textattack.attack_recipes as recipes
import os

# Import configurations
from config import MAX_LEN, TEXT_FEATURE, ADVERSARIAL_EXAMPLES_PREFIX, ADVERSARIAL_EXAMPLES_DIR

# Import tokenize_text from data_preparation for use in wrapper
from data_preparation import tokenize_text 

class HybridModelTextAttackWrapper(ModelWrapper):
    """
    Custom TextAttack ModelWrapper for a hybrid model.
    This wrapper assumes structured features are provided as a fixed tensor for all attacked texts.
    """
    def __init__(self, keras_model, tokenizer_obj, preprocessor_obj, structured_features_template):
        self.model = keras_model
        self.tokenizer = tokenizer_obj
        self.preprocessor = preprocessor_obj
        # structured_features_template should be a tf.Tensor of shape (1, structured_input_dim)
        # It serves as the baseline for the structured features input for TextAttack.
        self.structured_features_template = structured_features_template

    def __call__(self, text_input_list):
        """
        Predicts probabilities for a list of text inputs.
        The structured features are tiled from the template for batch prediction.
        """
        # Tokenize text inputs
        input_ids, attention_mask = tokenize_text(pd.Series(text_input_list), self.tokenizer, MAX_LEN)

        # Replicate the structured_features_template for the current batch size
        num_examples = len(text_input_list)
        batch_structured_features = tf.tile(self.structured_features_template, [num_examples, 1])

        # Make predictions using the hybrid model
        predictions = self.model({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structured_features_input': batch_structured_features
        })
        
        # TextAttack expects raw logits or probabilities for each class.
        # Our model outputs a single sigmoid probability for the positive class (phishing).
        # Convert it to a 2-element array [prob_class_0, prob_class_1]
        predictions = tf.concat([1 - predictions, predictions], axis=-1)
        
        return predictions.numpy()


def generate_adversarial_data(model_wrapper, tokenizer, preprocessor, X_source_data_df, y_source_data_tf, iteration_num, num_attacks_to_generate=50):
    """
    Generates successful adversarial examples from phishing messages in the source data.
    Saves and loads generated examples to/from a checkpoint file.
    
    Args:
        model_wrapper (HybridModelTextAttackWrapper): The TextAttack wrapper for the model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text.
        preprocessor (sklearn.compose.ColumnTransformer): The preprocessor for structured data.
        X_source_data_df (pd.DataFrame): The DataFrame containing original features (text and structured).
        y_source_data_tf (tf.Tensor): The TensorFlow tensor containing original labels.
        iteration_num (int or str): Current adversarial training iteration number (for checkpoint naming).
        num_attacks_to_generate (int): The maximum number of successful adversarial examples to generate.

    Returns:
        tuple: (list of adversarial texts, list of original structured feature Dataframes, list of original labels).
    """
    checkpoint_file = f"{ADVERSARIAL_EXAMPLES_PREFIX}{iteration_num}.pkl"

    if os.path.exists(checkpoint_file):
        print(f"Loading adversarial examples from checkpoint: {checkpoint_file}")
        try:
            loaded_data = pd.read_pickle(checkpoint_file)
            print(f"Loaded {len(loaded_data[0])} adversarial examples from checkpoint.")
            return loaded_data
        except Exception as e:
            print(f"Error loading adversarial examples from checkpoint: {e}. Regenerating...")
            # Fall through to generation if loading fails

    print(f"\n--- Generating up to {num_attacks_to_generate} Adversarial Messages from source data (Iteration {iteration_num}) ---")

    y_source_data_np = y_source_data_tf.numpy()
    
    # Get the actual index labels of phishing examples from X_source_data_df
    phishing_index_labels = X_source_data_df[y_source_data_np == 1].index.tolist()

    if not phishing_index_labels:
        print("No phishing examples found in source data for adversarial attack.")
        return [], [], []

    # Sample a subset of phishing examples to attack
    num_attacks = min(num_attacks_to_generate, len(phishing_index_labels))
    sampled_phishing_indices = np.random.choice(phishing_index_labels, num_attacks, replace=False)
    
    X_phishing_sampled = X_source_data_df.loc[sampled_phishing_indices]
    
    # TextAttack Dataset expects (text, label) pairs. Label 1 means positive class (phishing).
    textattack_dataset = Dataset([
        (row[TEXT_FEATURE], 1) # TextAttack expects original label for attack target
        for idx, row in X_phishing_sampled.iterrows()
    ])

    attack = recipes.TextFoolerJin2019.build(model_wrapper)

    attack_args = AttackArgs(
        num_examples=len(textattack_dataset),
        log_to_csv=os.path.join(ADVERSARIAL_EXAMPLES_DIR, f"log_textattack_iter_{iteration_num}.csv"), # Save detailed log
        disable_stdout=True # Suppress TextAttack's verbose stdout during progress
    )

    attacker = Attacker(attack, textattack_dataset, attack_args)
    results = attacker.attack_dataset()

    adversarial_texts = []
    adversarial_original_structured_features_df_list = []
    adversarial_labels = []

    # --- MODIFICATION START ---
    # Process the results more robustly to extract successful attacks
    for i, result in enumerate(results):
        if result.perturbed_text() is not None:
            # This is a successful attack if the perturbed text exists
            original_text = result.original_text()
            perturbed_text = result.perturbed_text()
            original_label = result.original_result.ground_truth_output
            
            # The order of results from TextAttack corresponds to the order in textattack_dataset
            original_row_idx = sampled_phishing_indices[i]
            original_structured_features_for_this_sms = X_source_data_df.loc[[original_row_idx]].drop(columns=[TEXT_FEATURE])
            
            adversarial_texts.append(perturbed_text)
            adversarial_original_structured_features_df_list.append(original_structured_features_for_this_sms)
            adversarial_labels.append(original_label) # Adversarial example keeps its original label
            
            # Print success message for visibility
            print(f"  SUCCESS! Original: '{original_text}' -> Adversarial: '{perturbed_text}'")
    # --- MODIFICATION END ---
    
    print(f"Generated {len(adversarial_texts)} successful adversarial examples.")
    
    # Save generated adversarial examples to checkpoint
    if adversarial_texts: # Only save if examples were generated
        # Ensure ADVERSARIAL_EXAMPLES_DIR exists
        os.makedirs(ADVERSARIAL_EXAMPLES_DIR, exist_ok=True)
        data_to_save = (adversarial_texts, adversarial_original_structured_features_df_list, adversarial_labels)
        pd.to_pickle(data_to_save, checkpoint_file)
        print(f"Saved adversarial examples to checkpoint: {checkpoint_file}")

    return adversarial_texts, adversarial_original_structured_features_df_list, adversarial_labels
