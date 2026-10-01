"""A simple CNN classifier for mel-spectrogram features.

Mel-spectrograms have a 2D structure (time x frequency), making them
suitable for convolutional neural networks.
"""

import torch.nn as nn


class EmotionCNN(nn.Module):
    """Convolutional network that predicts an emotion class from mel-spectrogram features.
    
    Since mel-spectrograms are 2D (time x frequency), we reshape the 1D feature
    vector into a 2D format and apply convolutions.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, n_mels=64):
        """Build the CNN layers.
        
        Args:
            input_size (int): Size of the input feature vector (should be n_mels).
            hidden_size (int): Number of channels in hidden layers.
            num_classes (int): Number of emotion classes to predict.
            n_mels (int): Number of mel frequency bands (used for reshaping).
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, hidden_size, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.3),
            
            # Second conv layer
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size*2),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.3),
            
            # Third conv layer (added to match the 3-layer recommendation)
            nn.Conv2d(hidden_size*2, hidden_size*4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size*4),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.3),
        )
        
        # After three max pooling with kernel_size=2, height becomes n_mels/8
        self.flattened_size = (hidden_size * 4) * (n_mels // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, feature_vector):
        """Run a forward pass.
        
        Args:
            feature_vector (torch.Tensor): Batch of feature vectors,
                shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: Raw class scores (logits), shape (batch_size, num_classes).
        """
        x = feature_vector.view(-1, 1, feature_vector.size(1), 1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x