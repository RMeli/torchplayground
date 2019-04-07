import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi Later Perceptron (MLP), aka fully connected neural network.
    """

    def __init__(self, layers_sizes, dropout_probability=None):

        super(MLP, self).__init__()

        self.layers_sizes = layers_sizes

        # Define list of layers (submodules)
        self.layers = nn.ModuleList()
        for in_size, out_size in zip(layers_sizes[:-2], layers_sizes[1:-1]):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.ReLU())

            # Add dropout layer
            if dropout_probability is not None:
                self.layers.append(nn.Dropout(dropout_probability))

        self.layers.append(nn.Linear(layers_sizes[-2], layers_sizes[-1]))

    def forward(self, x):
        # Reshape
        x = x.view(-1, self.layers_sizes[0])

        # Propagate
        for l in self.layers:
            x = l(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    model = MLP((28 * 28, 256, 64, 10), 0.25)

    print(model)
