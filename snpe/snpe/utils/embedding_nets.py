from typing import List

import torch


def get_cnn_1d(
    x: torch.Tensor,
    num_conv_layers: int = 4,
    num_channels: int = 8,
    conv_kernel_size: int = 5,
    maxpool_kernel_size: int = 5,
    num_dense_layers: int = 3,
) -> torch.nn.Module:
    # Get the input dimensionality of the first linear layer in the model
    # Depends on whether conv_kernel_size is odd or even
    # https://discuss.pytorch.org/t/how-can-i-ensure-that-my-conv1d-retains-the-same-shape-with-unknown-sequence-lengths/73647/8
    linear_input_dim = ((x.size()[-1] - ((conv_kernel_size + 1) % 2)) // maxpool_kernel_size) * num_channels

    # Build the modules that will make up the embedding CNN
    # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104/4
    cnn_modules = []  # type: List[torch.nn.Module]
    # First convolutional layer has 5 input channels - one for each star rating value
    cnn_modules.append(
        torch.nn.Conv1d(
            in_channels=5,
            out_channels=num_channels,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            dilation=1,
        )
    )
    for layer in range(1, num_conv_layers):
        cnn_modules.append(torch.nn.LeakyReLU())
        cnn_modules.append(
            torch.nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) * (2 ** layer) // 2,
                dilation=2 ** layer,
            )
        )
    cnn_modules.append(torch.nn.MaxPool1d(kernel_size=maxpool_kernel_size))
    # We put non-linearity after Max Pooling and Flattening the last layer as ReLU and MaxPooling commute
    # i.e, their MaxPool(Relu) = Relu(MaxPool) - the RHS is faster as MaxPool has reduced dimensions
    # https://stackoverflow.com/questions/35543428/activation-function-after-pooling-layer-or-convolutional-layer
    cnn_modules.append(torch.nn.Flatten(start_dim=1))
    cnn_modules.append(torch.nn.LeakyReLU())
    cnn_modules.append(torch.nn.Linear(linear_input_dim, 32 * (2 ** (num_dense_layers - 1))))
    for layer in range(1, num_dense_layers):
        # We will just ensure that the output dimensionality of the linear layers is fixed at 32
        cnn_modules.append(torch.nn.LeakyReLU())
        cnn_modules.append(
            torch.nn.Linear(32 * (2 ** (num_dense_layers - layer)), 32 * (2 ** (num_dense_layers - layer - 1)))
        )

    return torch.nn.Sequential(*cnn_modules)
