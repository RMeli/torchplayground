import torch

from matplotlib import pyplot as plt

from models import style
from utils.images import load_image, tensor_to_image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image and style image
img_name, style_name = "octopus", "hockney"
img = load_image("data/images/" + img_name + ".jpg").to(device)
style_img = load_image("data/images/" + style_name + ".jpg", shape=img.shape[2:]).to(device)

# Define style transfer object
st = style.StyleTransfer(img, style_img, device=device)

# Perform style transfer
target = st.run(2500)

plt.figure()
plt.imshow(tensor_to_image(target))
plt.savefig("data/images/" + img_name + '_' + style_name + ".png")
