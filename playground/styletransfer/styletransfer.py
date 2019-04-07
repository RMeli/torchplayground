"""
Implementation of style transfer using PyTorch

Notes
-----
Sources:
    [1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
    [1] Image Style Transfer Using Convolutional Neural Networks
    [1] https://arxiv.org/abs/1508.06576
    
    [2] Intro to Deep Learning with Pytorch
    [2] Udacity Course
    [2] https://www.udacity.com/course/deep-learning-pytorch--ud188

    [3] PyTorch Documentation
    [3] https://pytorch.org/docs/stable/index.html
"""

import torch
import torch.optim as optim

import torchvision

from typing import Dict, List, Tuple, Iterable


class StyleTransfer:
    """
    Style transfer class
    """

    def __init__(
        self,
        content: torch.tensor,  # Content image
        style: torch.tensor,  # Stile image
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """
        Style transfer __init__

        Parameters
        ----------
        self : StyleTransfer
        content : torch.tensor
           Content image (and initial guess)
        style : torch.tensor
            Style image
        device : torch.device
            Computation device (CPU or GPU)

        Notes
        -----
        Using a GPU us highly recommended, since the code is very slow on a CPU.

        One instance of this class corresponds to one style transfer between the style
        image and the content image (the content is used as target). The transfer can be
        performed multiple times with different parameters using the function `run()`.
        """
        self.content = content
        self.style = style
        self.device = device

        # Load/get VGG19 CNN
        self.vgg = self._load_vgg19()

        # Get content and style features
        self.content_features = self._get_features(self.content)
        self.style_features = self._get_features(self.style)

        # Compute Gram matrix for each layer of the style features
        self.style_grams = {
            layer: self._gram_matrix(self.style_features[layer])
            for layer in self.style_features
        }

    def _load_vgg19(self) -> torch.nn:
        """
        Load VGG19 neural network

        Returns
        -------
        vgg : torch.model
            VGG19 convolutional layers (pre-trained)

        Notes
        -----
        VGG19 is a pre-trained CNN downloaded from torchvision.
        Downloading the model can take some time and requires internet connection.
        """

        # Load convolutional ("features") layers of VGG19
        # The fully connected layers ("classifier") are not needed
        vgg = torchvision.models.vgg19(pretrained=True).features

        # Freeze all the pre-trained VGG19 parameters
        for p in vgg.parameters():
            p.requires_grad_(False)

        # Move the model to the correct device
        return vgg.to(self.device)

    def _get_features(self, image: torch.tensor) -> Dict[str, torch.tensor]:
        """
        Get selected features layers from VGG19 for a specific image passed
        through the network.

        Parameters
        ----------
        image : torch.tensor
            Current image, to be propagated through the network
        
        Returns
        -------
        features : Dict[torch.tensor]
            Features of the input images for specific convolutional layers
            (both for style and content)

        Notes
        -----
        The features (convolutional layers) are selected according to Gatys et al. (2016).

        VGG19 structure in PyTorch:
            VGG(
                (features): Sequential(
                    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (1): ReLU(inplace)
                    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (3): ReLU(inplace)
                    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (6): ReLU(inplace)
                    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (8): ReLU(inplace)
                    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (11): ReLU(inplace)
                    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (13): ReLU(inplace)
                    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (15): ReLU(inplace)
                    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (17): ReLU(inplace)
                    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (20): ReLU(inplace)
                    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (22): ReLU(inplace)
                    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (24): ReLU(inplace)
                    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (26): ReLU(inplace)
                    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (29): ReLU(inplace)
                    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (31): ReLU(inplace)
                    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (33): ReLU(inplace)
                    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (35): ReLU(inplace)
                    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                )
                (classifier): Sequential(
                    (0): Linear(in_features=25088, out_features=4096, bias=True)
                    (1): ReLU(inplace)
                    (2): Dropout(p=0.5)
                    (3): Linear(in_features=4096, out_features=4096, bias=True)
                    (4): ReLU(inplace)
                    (5): Dropout(p=0.5)
                    (6): Linear(in_features=4096, out_features=1000, bias=True)
                )
            )  
        """

        # VGG19 layers, according to Gatys et al. (2016)
        # See above for the PyTorch structure
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Content layer
            "28": "conv5_1",
        }

        # Features (CNN layers) for current images
        features = {}

        # Get features for current image
        x = image
        for name, layer in self.vgg._modules.items():
            # Propagate the image through the network (layer-by-layer)
            x = layer(x)

            # Save current layer if needed
            if (
                name in layers
            ):  # PyTorch ame is the index (key in the dictionary layers)
                features[layers[name]] = x

        return features

    def _gram_matrix(self, tensor: torch.tensor) -> torch.tensor:
        """
        Compute Gram matrix for a given tensor

        Parameters
        ----------
        tensor : torch.tensor
            Input tensor
        
        Returns
        -------
        gram : torch.tensor
            Gram matrix of the input tensor

        Notes
        -----
        The Gram matrix of a set of vectors $v_1, \dots, v_n$ is the matrix of inner products
        whose elements are given by
        $$
            G_{ij} = \angle v_i, v_j \rangle.
        $$
        If $V$ is the matrix whose columns are the vectors $v_1, \dots, v_n$, we have
        $$
            G = V^TV.
        $$

        The Gram matrix corresponds to Equation 3 in Gatys et al. (2016).
        """

        # Get depth (number of channels), height and width of current tensor
        # Discard batch size
        _, d, h, w = tensor.size()

        # Reshape tensor to matrix
        tensor = tensor.view(d, h * w)

        # Compute Gram matrix
        return torch.mm(tensor, tensor.t())

    def _optimizer_factory(
        self, params: Iterable[torch.tensor], optimizer: str, lr: float
    ) -> torch.optim.Optimizer:
        """
        Optimizer factory function

        Parameters
        ----------
        params: Iterable[torch.tensor]
            Parameters to be optimized
        optimizer: str
            Name of the optimizer (SGD, Adam, LBFGS)
        lr : float
            Learning rate

        Return
        ------
        optimizer: torch.optim.Optimizer
            Optimizer

        Note
        ----
        LBFGS is quite slow and memory intensive. However it works best for style transfer,
        according to Gatys et al. (2016).
        """

        optimizers = {
            "sgd": optim.SGD([params], lr=lr),
            "adam": optim.Adam([params], lr=lr),
            "lbfgs": optim.LBFGS([params], lr=lr, history_size=50),
        }

        try:
            return optimizers[optimizer.lower()]
        except KeyError:
            print("Unknown optimizer. Setting optimizer to LBFGS.")
            return optimizers["lbfgs"]

    def run(
        self,
        steps: int,
        content_weight: float = 1,  # Conetnt weight for total loss
        style_weight: float = 1e3,  # Style weight for total loss
        style_layers_weights: List[float] = [0.2] * 5,  # Weights for style layers
        optimizer_name="LBFGS",  # LBFGS optimizer
        lr: float = 1.0,
    ) -> torch.tensor:
        """
        Perform style transfer with the specified parameters

        Parameters
        ----------
        steps : int
            Total number of optimization (transfer) steps
        content_weight : float
            Conetnt weight for total loss
        style_weight : float
            Style weight for total loss
        style_layer_weights : Dict[str, float]
            Style weight for VGG19 layers using in style transfer
        optimizer_name : torch.optim.Optimizer
            Name of the optimizer
        lr : float
            Learning rate for the optimizer

        Returns
        -------
        image : torch.tensor
            Image after style transfer
        """
        # Create target image
        # The target is initialized as content (and iteratively modified to change the style)
        target = self.content.clone()
        target.requires_grad_(True)  # Activate gradients calculation
        target = target.to(self.device)  # Move to device

        # Map style layer weights to names
        style_layer_names = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        style_weights = {
            name: weight
            for name, weight in zip(style_layer_names, style_layers_weights)
        }

        # Set optimizer
        optimizer = self._optimizer_factory(target, optimizer_name, lr)

        for _ in range(steps):

            def closure():
                """
                Closure for optimization
                """

                # Reset gradients
                optimizer.zero_grad()

                # Get target features
                # This call performs a forward propagation of the target image
                target_features = self._get_features(target)

                # Compute content loss (conv4_2 is the content layer)
                # Equation 1 in Gatys et al. (2016)
                content_loss = 0.5 * torch.sum(
                    (target_features["conv4_2"] - self.content_features["conv4_2"]) ** 2
                )

                # Compute style loss
                style_loss = 0
                for layer in style_weights:
                    # Get target feature for current layer
                    target_feature = target_features[layer]

                    # Compute Gram matrix for target
                    # Matrix G in Equation 4 of Gatys et al. (2016)
                    target_gram = self._gram_matrix(target_feature)

                    # Get Gram matrix for style (current layer)
                    # Matrix A in Equation 4 of Gatys et al. (2016)
                    style_gram = self.style_grams[layer]

                    # Compute the style loss for current layer
                    # Equation 4 in Gatys et al. (2016)
                    _, d, h, w = target_feature.shape
                    layer_style_loss = torch.mean((target_gram - style_gram) ** 2) / (
                        4 * d * h * w
                    )

                    # Update total style loss
                    # Equation 5 in Gatys et al. (2016)
                    style_loss += style_weights[layer] * layer_style_loss

                # Total loss
                # Equation 7 in Gatys et al. (2016)
                loss = content_weight * content_loss + style_weight * style_loss

                # Backpropagation
                loss.backward()

                return loss

            # Update target
            optimizer.step(closure)

        return target.clone().detach()


if __name__ == "__main__":
    import argparse
    import numpy as np

    from matplotlib import pyplot as plt
    from PIL import Image

    from torchvision import transforms

    def get_args():
        """
        Parse arguments from command line
        """
        parser = argparse.ArgumentParser(
            description="Neural Style Transfer using PyTorch"
        )

        parser.add_argument(
            "--style",
            "-s",
            type=str,
            required=True,
            help="Style image",
            metavar="STYLE_IMAGE",
        )
        parser.add_argument(
            "--content",
            "-c",
            type=str,
            required=True,
            help="Content image",
            metavar="CONTENT_IMAGE",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            required=True,
            help="Output image",
            metavar="OUTPUT_IMAGE",
        )
        parser.add_argument(
            "--nsteps",
            "-n",
            type=int,
            required=True,
            help="Number of steps",
            metavar="STEPS",
        )
        parser.add_argument(
            "--alpha", "-a", type=float, default=1, help="Content loss weight (alpha)"
        )
        parser.add_argument(
            "--beta", "-b", type=float, default=1e6, help="Style loss weight (beta)"
        )
        parser.add_argument(
            "--weights",
            "-w",
            type=float,
            nargs=5,
            default=[0.2] * 5,
            help="Style layers weights",
            metavar="WEIGHT",
        )
        parser.add_argument(
            "--optimizer", "-z", type=str, default="lbfgs", help="Optimizer"
        )
        parser.add_argument(
            "--lr", "-l", type=float, default=0.002, help="Learning rate"
        )

        return parser.parse_args()

    def load_image(img_path: str, shape: Tuple[int] = None) -> torch.tensor:
        """
        Transform image to torch.tensor (with normalisation)

        Parameters
        ----------
        img_path : str
            Path to image
        shape : Tuple[int]
            Image shape (for respahing/resizing)

        Returns
        -------
        img : torch.tensor
            Image as torch.tensor
        """

        # Load image in RGB format
        img = Image.open(img_path).convert("RGB")

        # Image transformations
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))

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

    def tensor_to_image(tensor: torch.tensor) -> np.ndarray:
        """
        Transform torch.tensor to RGB image

        Parameters
        ----------
        tensor : torch.tensor
            Image as torch.tensor

        Returns
        -------
        img : np.ndarray
            Image as np.array

        Note
        ----
        For RGB images, PIL expects a tensor of integers (between 0 and 255) with shape (channels, h, w)
        """

        # Move tensor to CPU
        img = tensor.to("cpu")

        # Transform image into numpy array
        img = img.numpy().squeeze().transpose(1, 2, 0)

        # Un-normalize
        img = img * np.array((0.2, 0.2, 0.2)) + np.array((0.5, 0.5, 0.5))

        # Clip for plt.imgshow() (to avoid warning)
        img = img.clip(0, 1)

        return img

    # Select device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    args = get_args()

    # Load content and style images
    img_content = load_image(args.content).to(device)
    img_style = load_image(args.style, shape=img_content.shape[2:]).to(device)

    # Setup style transfer
    st = StyleTransfer(img_content, img_style)

    # Run style transfer
    img = st.run(
        args.nsteps,
        content_weight=args.alpha,
        style_weight=args.beta,
        style_layers_weights=args.weights,
        optimizer_name=args.optimizer,
        lr=args.lr,
    )

    # Save output image
    plt.figure()
    plt.axis("off")
    plt.imshow(tensor_to_image(img))
    plt.savefig(args.output)
