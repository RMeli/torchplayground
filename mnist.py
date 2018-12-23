import torch
from torchvision import datasets, transforms
from torch.utils.data import sampler

import numpy as np
from matplotlib import pyplot as plt

# Define data transformation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Download train and test sets
train_set = datasets.MNIST(
    root="data/MNIST", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    root="data/MNIST", train=False, download=True, transform=transform
)

# Obtain size
num_train, num_test = len(train_set), len(test_set)

# Check size
assert num_train == 60000
assert num_test == 10000

# Obtain train and validation indices dfrom train set
validation_ratio = 0.2
split = int(np.floor(validation_ratio * num_train))
idx = list(range(num_train))
np.random.shuffle(idx)  # Shuffle idx in-place
train_idx, validation_idx = idx[split:], idx[:split]
num_train, num_validation = len(train_idx), len(validation_idx)

# Check train and validation split
assert num_train + num_validation == len(train_set)
assert int(np.floor(len(train_set) * validation_ratio)) == num_validation
assert int(np.floor(len(train_set) * (1 - validation_ratio))) == num_train

# Define sampler for train and validation sets
train_sampler = sampler.SubsetRandomSampler(train_idx)
validation_sampler = sampler.SubsetRandomSampler(validation_idx)

# Define data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, sampler=validation_sampler
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# Load a batch of train images (and labels)
train_iterator = iter(train_loader)
images, labels = train_iterator.next()
images = images.numpy()

# Check image size
assert images.shape == (batch_size, 1, 28, 28)

# Plot batch of train images
fig = plt.figure()
for idx in range(batch_size):
    n = np.rint(np.sqrt(batch_size))
    ax = fig.add_subplot(n, n, idx + 1, xticks=[], yticks=[])
    plt.imshow((images[idx][0] + 0.5) * 0.5)  # Un-normalise
    ax.set_title(labels[idx].item(), fontdict={"fontsize": 6})
plt.tight_layout()
plt.show()
