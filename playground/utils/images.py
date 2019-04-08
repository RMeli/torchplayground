from torchvision import transforms

import numpy as np

from PIL import Image


def load_image(img_path, shape=None):

    # Load image in RGB format
    img = Image.open(img_path).convert("RGB")

    # Image transformations
    # The normalization parameters are chosen according to PyTorch documentation:
    # https://pytorch.org/docs/stable/torchvision/models.html
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if shape is None:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(shape), transforms.ToTensor(), normalize]
        )

    # Remove alpha channel and add batch dimensions
    # Shape: (Batch, Channel, X, Y)
    img = transform(img)[:3, :, :].unsqueeze(0)

    return img


def tensor_to_image(tensor):

    # Move tensor to CPU
    img = tensor.to("cpu").clone().detach()

    # Transform image into numpy array
    img = img.numpy().squeeze().transpose(1, 2, 0)

    # Un-normalize
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    # Clip for plt.imgshow() (to avoid warning)
    img = img.clip(0, 1)

    return img
