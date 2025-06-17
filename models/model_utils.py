"""
Shared model utilities for the credit scoring experiment
"""

import numpy as np
from typing import Any, Callable, Dict, List, Tuple

class TemperatureScaledModel:
    """
    Model wrapper for temperature scaling calibration
    """
    def __init__(self, original_model, temperature, get_model_logits_func, temperature_scaling_func):
        self.original_model = original_model
        self.temperature = temperature
        self._get_model_logits = get_model_logits_func
        self._temperature_scaling = temperature_scaling_func
        
    def predict_proba(self, X):
        test_logits = self._get_model_logits(self.original_model, X)
        calibrated_probs = self._temperature_scaling(test_logits, self.temperature)
        return np.column_stack([1 - calibrated_probs, calibrated_probs])
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
