# -*-coding:utf-8-*-
import torch

from models.networks.cnn import NeuralNetwork
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.NeuralNetwork = NeuralNetwork()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        assert list(inputs.shape[-2:]) == [28, 28]
        outputs = self.NeuralNetwork(x=inputs)
        loss = self.loss(input=outputs, target=targets)
        return {"outputs": outputs, "loss": loss}
