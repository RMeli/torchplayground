import torch
import torch.optim as optim
from torch.utils.data import sampler

from torchvision import datasets, transforms

import numpy as np
import os
from matplotlib import pyplot as plt

import mnist.mlp as mlp
from utils.train import train
from utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=64

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

# Check size
assert len(train_set) == 60000
assert len(test_set) == 10000

train_loader, validation_loader = data.split_train_and_validation(train_set, batch_size)

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

# Define the model
model = mlp.MLP((28 * 28, 512, 512, 10), 0.25)

# Move model to device
model = model.to(device)

# Set optimizer and bind to model
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train model
epochs = 10
model_name = "data/MNIST/mlp_mnist.pt"
loss = nn.NLLLoss()
if os.path.isfile(model_name):
    # Load the model
    model.load_state_dict(torch.load(model_name))
else:
    train_loss, validation_loss = train(
        epochs,
        model,
        loss,
        optimizer,
        train_loader,
        validation_loader,
        model_name,
        device,
    )

    # Show train and validation losses
    fig = plt.figure()
    plt.plot(range(epochs), train_loss, label="Train Loss")
    plt.plot(range(epochs), validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

# Test the model
test_loss = 0.0
class_correct = list(range(10))
class_total = list(range(10))
model.eval()
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    predicted = model(data)
    l = loss(predicted, target)
    test_loss += l.item() * data.size(0)
    _, pred = torch.max(predicted, dim=1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(len(target.data)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
test_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_loss:.6f}")
for i in range(10):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        num_correct = np.sum(class_correct[i])
        num_total = np.sum(class_total[i])
        print(f"Test Accuracy of {i}: {accuracy:.2f}% ({num_correct}/{num_total})")
    else:
        print(f"Test Accuracy of {classes[i]}: N/A")

# Load a batch of test images (and labels) and compute predictions
model = model.to("cpu")
test_iterator = iter(test_loader)
images, labels = test_iterator.next()
predicted = model(images)
_, preds = torch.max(predicted, dim=1)
images = images.numpy()

# Plot batch of test images
fig = plt.figure()
for idx in range(batch_size):
    n = np.rint(np.sqrt(batch_size))
    ax = fig.add_subplot(n, n, idx + 1, xticks=[], yticks=[])
    plt.imshow((images[idx][0] + 0.5) * 0.5)  # Un-normalise
    ax.set_title(
        f"{preds[idx].item()} ({labels[idx].item()})",
        fontdict={"fontsize": 6},
        color=("green" if preds[idx] == labels[idx] else "red"),
    )
plt.tight_layout()
plt.show()
