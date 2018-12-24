import torch

import numpy as np

def train(epochs, model, loss, optimizer, train_loader, validation_loader=None, model_name="model.pt"):

    train_loss = []
    validation_loss = []

    validation_loss_min = np.Inf

    for epoch in range(epochs):
        current_train_loss = 0
        current_validation_loss = 0

        # Model training
        model.train()
        for data, target in train_loader:
            # Initialize gradients
            optimizer.zero_grad()

            # Model evaluation (forward propagation)
            prediction = model(data)

            # Loss
            l = loss(prediction, target)

            # Backpropagation
            l.backward()

            # Optimization step
            optimizer.step()

            # Update training loss
            current_train_loss += l.item() * data.size(0)
        
        # Model validation
        if validation_loader is not None:
            model.eval()

            for data, target in validation_loader:
                # Model evaluation (forward propagation)
                prediction = model(data)

                # Loss
                l = loss(prediction, target)

                # Update validation loss
                current_validation_loss += l.item() * data.size(0)

        train_loss.append(current_train_loss / len(train_loader.dataset))
        if validation_loader is None:
            print(f"Epoch: {epoch}\n    Training Loss: {train_loss[-1]:.6f}")
        else:
            validation_loss.append(current_validation_loss / len(validation_loader.dataset))
            print(f"Epoch: {epoch}\n    Training Loss: {validation_loss[-1]:.6f}    Validation Loss: {validation_loss[-1]:.6f}")

        if validation_loss[-1] < validation_loss_min:
            print(f"        Validation loss decreased. Saving model...")
            torch.save(model.state_dict(), model_name)
            validation_loss_min = validation_loss[-1]

    if validation_loader is None:
        return train_loss
    else:
        return train_loss, validation_loss


    



