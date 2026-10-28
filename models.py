import torch
import torch.nn as nn
from morph_pool_layer import AdaptiveMorphPooling2d

class CNNClassifier(nn.Module):
    def __init__(self, use_morph_pool=False, metric='SMO', stat_window = None):
        super(CNNClassifier, self).__init__()
        
        self.use_morph_pool = use_morph_pool

        # Submuestreo inicial sobre la imagen RGB (3 canales)
        if use_morph_pool:
            self.morph_pool = AdaptiveMorphPooling2d(kernel_size=2, stride=2, metric=metric, stat_window= stat_window)
        else:
            self.max_pool_rgb = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloques Convolucionales
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Primero reduce la resolución en RGB
        if self.use_morph_pool:
            x = self.morph_pool(x)
        else:
            x = self.max_pool_rgb(x)
            
        # Pasa las imágenes reducidas a la CNN
        x = self.features(x)
        logits = self.classifier(x)
        return logits