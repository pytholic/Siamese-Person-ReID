from dataclasses import dataclass


@dataclass
class Args:
    """
    Training arguments.
    """

    # Learning rate for the optimizer
    learning_rate: float = 1e-3
    # Training batch size
    batch_size: int = 16
    # Maximum number of training epochs
    epochs: int = 1  # 100
    # Validation interval
    val_interval = 1
    
    # Early stopping args
    patience: int = 5  # Number of epochs with no improvement after which training will be stopped
    min_delta: float = 0.001  # Minimum change in validation loss to be considered as an improvement
