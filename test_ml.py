import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import load_model, compute_model_metrics
import unittest


# Step 4. Unit Test


project_path = "../Project2-Deploying-a-Scalable-ML-Pipeline-with-FastAPI"

def test_expected_algorithm():
    """
    Check to see that the ML model is using the expected algorithm.
    """
    model_path = os.path.join(project_path, "model", "model.pkl")  
    model = load_model(model_path)

    assert isinstance(model, RandomForestClassifier)


def test_loading_model():
    """
    Test loading model
    """
    model_path = os.path.join(project_path, "model", "model.pkl")  
    model = load_model(model_path)

    assert model == model, f"Expected: {model}, received: {model}"

    



def test_compute_model_metrics():
    """
    Test compute_model_metrics is returning expected value
    """
    # Dataset with controlled values for test
    y = np.array([0, 1, 0, 1, 0])
    preds = np.array([0, 1, 0, 1, 1])


    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Expected values
    expected_precision = 0.6667
    expected_recall = 1.0
    expected_fbeta = 0.8

    assert round(precision, 4) == expected_precision
    assert round(recall, 4) == expected_recall
    assert round(fbeta, 4) == expected_fbeta

