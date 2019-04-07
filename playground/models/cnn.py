import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Simple CNN with convolutional and fully connected layers
    """

    def __init__(self, cnn_layers, fc_layers, kernel_size=3):

        super(CNN, self).__init__()

        # Avoid complications with padding
        if kernel_size % 2 == 0:
            raise ValueError("Only odd kernel sizes supported.")
        else:
            pad = (kernel_size - 1) // 2

        self.cnn = nn.ModuleList()

        for size_in, size_out in zip(cnn_layers[:-1], cnn_layers[1:]):
            self.cnn.append(nn.Conv2d(size_in, size_out, kernel_size=kernel_size ,padding=pad))
            self.cnn.append(nn.MaxPool2d(2, 2))

        self.fc = nn.ModuleList()
        for size_in, size_out in zip(fc_layers[:-2], fc_layers[1:-1]):
            self.fc.append(nn.Linear(size_in, size_out))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(fc_layers[-2], fc_layers[-1]))


    def forward(self, x):

        # Propagate through convolutional layers
        for l in self.cnn:
            x = l(x)

        # Reshape
        x = x.view(-1, self.fc[0].in_features)

        # Propagate through fully connected laters
        for l in self.fc:
            x = l(x)

        return F.log_softmax(x, dim=1)