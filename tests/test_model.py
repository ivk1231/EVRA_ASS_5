import pytest
import torch
from model.mnist_model import CompactMNIST
from train import train, count_parameters

def test_model_parameters():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    accuracy, _ = train()
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%" 