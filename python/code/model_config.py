from dataclasses import dataclass
import numpy as np


@dataclass
class ModelTrainingParams:
    X_train_structured_processed: np.ndarray
    X_val_structured_processed: np.ndarray
    X_test_structured_processed: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_input_ids: np.ndarray
    train_attention_mask: np.ndarray
    train_token_type_ids: np.ndarray
    val_input_ids: np.ndarray
    val_attention_mask: np.ndarray
    val_token_type_ids: np.ndarray
    test_input_ids: np.ndarray
    test_attention_mask: np.ndarray
    test_token_type_ids: np.ndarray