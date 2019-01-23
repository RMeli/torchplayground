import torch
import torch.optim as optim

import torchvision
from torchvision import transforms


class StyleTransfer:
    """
    Style transfer.

    Source:
        Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
        Image Style Transfer Using Convolutional Neural Networks
    """

    def __init__(
        self,
        content,
        style,
        content_weight=1,
        style_weight=1e6,
        style_weights={
            "conv1_1": 1.0,
            "conv2_1": 0.75,
            "conv3_1": 0.2,
            "conv4_1": 0.2,
            "conv5_1": 0.2,
        },
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.content = content
        self.style = style
        self.content_weight = content_weight  # alpha
        self.style_weight = style_weight  # beta
        self.style_weights = style_weights
        self.device = device

        self.vgg = self._load_vgg19()
        
        # Get content and style features
        self.content_features = self._get_features(self.content)
        self.style_features = self._get_features(self.style)

        # Compute Gram matrix for each layer of the style features
        self.style_grams = {layer: self._gram_matrix(self.style_features[layer]) for layer in self.style_features}

        # Create target image
        # The target is initialized as content (and iteratively modified to change the style)
        self.target = self.content.clone()
        self.target.requires_grad_(True) # Activate gradients calculation
        self.target = self.target.to(device) # Move to device

    def _load_vgg19(self):
        """
        Load VGG19 neural network.
        """

        # Load convolutional ("features") layers of VGG19
        # The fully connected layers ("classifier") are not needed
        vgg = torchvision.models.vgg19(pretrained=True).features

        # Freeze all the pre-trained VGG19 parameters
        for p in vgg.parameters():
            p.requires_grad_(False)

        return vgg.to(self.device)

    def _get_features(self, image):
        # VGG19 layers from Gatys et al (2016)
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Content
            "28": "conv5_1",
        }

        # Features (CNN layers) for current images
        features = {}

        # Get features for current image
        x = image
        for name, layer in self.vgg._modules.items():
            # Propagate the image through the network (layer-by-layer)
            x = layer(x)

            if name in layers:
                features[layers[name]] = x

        return features

    def _gram_matrix(self, tensor):
        """
        Compute Gram matrix for given tensor.
        """

        # Get depth (number of channels), height and width of current tensor
        # Discard batch size
        _, d, h, w = tensor.size()

        # Reshape tensor to matrix
        tensor = tensor.view(d, h * w)

        return torch.mm(tensor, tensor.t())

    def run(self, steps=2500, save_every=500):

        images = []

        optimizer = optim.Adam([self.target], lr=0.002)

        for i in range(steps):

            # Get target features
            target_features = self._get_features(self.target)

            # Compute content loss 
            content_loss = torch.mean((target_features["conv4_2"] - self.content_features["conv4_2"])**2)

            # Compute style loss
            style_loss = 0
            for layer in self.style_weights:
                # Get target feature for current layer
                target_feature = target_features[layer]

                # Compute Gram matrix for target
                target_gram = self._gram_matrix(target_feature)

                # Get Gram matrix for style (current layer)
                style_gram = self.style_grams[layer]

                # Compute the style loss for current layer
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram)**2)

                # Update total style loss
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)

            # Total loss
            loss = self.content_weight * content_loss + self.style_weight * style_loss

            # Update target
            optimizer.zero_grad()
            loss.backward() # Backpropagation
            optimizer.step()

            if i % save_every == 0:
                images.append(self.target.clone())

        return self.target.clone().detach(), images

