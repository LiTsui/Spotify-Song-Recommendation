import torch.nn as nn

"""
Stores the Neural Network
"""


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # neural network layers
        self.net = nn.Sequential(
            nn.Linear(22, 100),  # 22 columns to 100 calculated values (neurons)
            nn.BatchNorm1d(100),  # normalize all 100 neurons
            nn.LeakyReLU(),  # lets negative values be used
            nn.Dropout(0.75),  # randomly disable 75% of the 100 neurons to reduce overfitting
            nn.Linear(100, 32),  # compress 100 to 32 neurons
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),  # outputs a number from -inf to inf
            nn.Sigmoid(),  # converts number to a probability in range [0, 1]
        )

    def forward(self, x):
        return self.net(x)