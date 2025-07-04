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
    
def generate_and_evaluate_adversarial_examples(trained_model, tokenizer, preprocessor, X_data, y_data):
    print("\n--- Phase 4: Adversarial Message Generation ---")

    # --- 1. Identify Phishing Examples for Attack ---
    phishing_indices = y_data[y_data == 1].index
    # Limit to a reasonable number for TextAttack, as it can be slow
    num_attacks = 50 # Adjust as needed for computation time
    if len(phishing_indices) > num_attacks:
        # Randomly sample phishing messages to attack
        sampled_phishing_indices = np.random.choice(phishing_indices, num_attacks, replace=False)
    else:
        sampled_phishing_indices = phishing_indices
    
    X_phishing_sampled = X_data.loc[sampled_phishing_indices]
    y_phishing_sampled = y_data.loc[sampled_phishing_indices]

    print(f"\nAttacking {len(sampled_phishing_indices)} phishing examples...")

    # Determine structured input dimension for the wrapper's placeholder
    # This requires processing at least one structured example
    sample_structured_features = preprocessor.transform(X_phishing_sampled.head(1)).toarray()
    structured_input_dim = sample_structured_features.shape[1]
    
    # Create a 'neutral' structured feature template (e.g., zeros or mean values)
    # This template will be used for all TextAttack runs, so TextAttack focuses on text.
    # For more realistic attacks, you'd use the original structured features for each example.
    # For now, let's use zeros for simplicity, or you could pre-calculate mean for phishing structured features.
    structured_features_template = tf.zeros((1, structured_input_dim), dtype=tf.float32)
    # Alternatively, use the actual structured features for the first example in the batch:
    # structured_features_template = tf.constant(preprocessor.transform(X_phishing_sampled.iloc[[0]]).toarray(), dtype=tf.float32)


    # --- 2. Text-level Attacks (using TextAttack) ---
    print("\n--- Running Text-level Adversarial Attacks ---")

    # Create TextAttack wrapper for our hybrid model
    attack_model_wrapper = HybridModelTextAttackWrapper(
        keras_model=trained_model,
        tokenizer_obj=tokenizer,
        preprocessor_obj=preprocessor,
        structured_features_template=structured_features_template
    )

    # Prepare TextAttack dataset: (text, label) pairs
    # TextAttack expects labels 0 for benign, 1 for attack target.
    # Since we are attacking phishing (label 1) to make it benign (target 0), it's consistent.
    textattack_dataset = Dataset([
        (row[1]['sms_content_cleaned_for_nlp'], 1) # TextAttack expects (text, label), label 1 means positive class (phishing)
        for row in X_phishing_sampled.iterrows()
    ])

    # Choose an attack recipe
    # You can experiment with different recipes or build custom ones
    # recipes.TextAttackWordNet uses WordNet for synonym replacement
    # recipes.TextAttackBART uses BART for paraphrasing (slower, needs BART model)
    # recipes.TextAttackDeepWordBug uses character-level transformations
    
    # Let's start with a simpler recipe like WordNet for demonstration
    attack = recipes.WordNetHomoglyphAttack.build(attack_model_wrapper) # Changed to WordNetHomoglyphAttack for faster demo and relevant example
    # This attack uses synonym replacement and character homoglyphs

    # Configure attack arguments
    attack_args = AttackArgs(
        num_examples=len(textattack_dataset), # Number of examples to attack
        log_to_csv="log_textattack.csv",
        log_to_stdout=True,
        log_to_file=True,
        disable_stdout=False
    )

    attacker = Attacker(attack, textattack_dataset, attack_args)
    results = attacker.attack_dataset()

    adversarial_text_examples = []
    attack_success_count_text = 0

    for result in results:
        # A successful attack means the perturbed text fooled the model (predicted a different class)
        # For a target label 1 (phishing), if the model predicts 0 (ham) after perturbation, it's successful.
        original_text = result.original_result.attacked_text.text
        perturbed_text = result.perturbed_result.attacked_text.text if result.perturbed_result else None
        original_score = result.original_result.score
        perturbed_score = result.perturbed_result.score if result.perturbed_result else None
        
        # Original label is 1 (phishing). We want the model to predict 0 (ham).
        # The scores from TextAttackWrapper are [prob_ham, prob_phishing]
        original_pred_class = np.argmax(original_score)
        perturbed_pred_class = np.argmax(perturbed_score) if perturbed_score is not None else original_pred_class

        # Check if the attack was successful (model changed prediction from phishing to ham)
        is_successful_attack = (original_pred_class == 1 and perturbed_pred_class == 0)

        adversarial_text_examples.append({
            'original_sms': original_text,
            'adversarial_sms': perturbed_text,
            'original_prediction': f'Phishing (score: {original_score[1]:.4f})',
            'adversarial_prediction': f'Ham (score: {perturbed_score[1]:.4f})' if perturbed_score else 'No change',
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

