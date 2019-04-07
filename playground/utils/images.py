from torchvision import transforms

import numpy as np

from PIL import Image


def load_image(img_path, shape=None, norm=(0.5, 0.5, 0.5), std=(0.2,0.2,0.2)):

    # Load image in RGB format
    img = Image.open(img_path).convert("RGB")

    # Image transformations
    normalize = transforms.Normalize(norm, std)

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


def tensor_to_image(tensorn, norm=(0.5, 0.5, 0.5), std=(0.2,0.2, 0.2)):

    # Move tensor to CPU
    img = tensor.to("cpu")

    # Transform image into numpy array
    img = img.numpy().squeeze().transpose(1, 2, 0)

    # Un-normalize
    img = img * np.array(std) + np.array(norm)

    # Clip for plt.imgshow() (to avoid warning)
    img = img.clip(0, 1)

    return img
