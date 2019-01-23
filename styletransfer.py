import torch

from matplotlib import pyplot as plt

from models import style
from utils.images import load_image, tensor_to_image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image and style image
img = load_image("data/images/octopus.jpg").to(device)
style_img = load_image("data/images/hockney.jpg", shape=img.shape[2:]).to(device)

# Show image and style image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(tensor_to_image(img))
ax[1].imshow(tensor_to_image(style_img))
plt.show()

# Define style transfer object
st = style.StyleTransfer(img, style_img)

# Perform style transfer
target, targets = st.run(50, 500)

plt.imshow(tensor_to_image(target))
plt.savefig("data/images/style_transfer.png")
plt.show()