import numpy as np

import torch
from torch.utils.data import sampler

def split_train_and_validation(train_set, batch_size, validation_ratio=0.2):
    # Train set size
    num_train = len(train_set)

    # Obtain train and validation indices from train set
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

    return train_loader, validation_loader
