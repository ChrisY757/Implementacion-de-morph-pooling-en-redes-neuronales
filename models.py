import torch
import torch.nn as nn
from morph_pool_layer import AdaptiveMorphPooling2d
import torchvision.models as models

class CNNClassifier(nn.Module):
    def __init__(self, use_morph_pool=False, metric='SMO', stat_window=None, pool_type='max'):
        super().__init__()
        self.use_morph_pool = use_morph_pool
        if use_morph_pool:
            self.pool = AdaptiveMorphPooling2d(2, 2, metric=metric, stat_window=stat_window)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)
        self.morph_pool = self.pool
        self.max_pool_rgb = self.pool

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
        x = self.pool(x)
        x = self.features(x)
        return self.classifier(x)


class PretrainedWithPool(nn.Module):
    def __init__(self, backbone_name='resnet18', use_morph=False, metric='SMO', stat_window=None, pool_type='max'):
        super().__init__()
        self.backbone_name = backbone_name
        if use_morph:
            self.pool = AdaptiveMorphPooling2d(2, 2, metric=metric, stat_window=stat_window)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)

        if backbone_name == 'vgg16':
            bb = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            for p in bb.parameters(): p.requires_grad = False
            for i in range(17, len(bb.features)):
                for p in bb.features[i].parameters(): p.requires_grad = True
            self.frozen_features = bb.features[:17].eval()
            self.trainable_features = bb.features[17:]
            self.features = bb.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(nn.Linear(512*7*7, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        elif backbone_name == 'resnet18':
            bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            for p in bb.parameters(): p.requires_grad = False
            for p in bb.layer3.parameters(): p.requires_grad = True
            for p in bb.layer4.parameters(): p.requires_grad = True
            self.frozen_features = nn.Sequential(
                bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2
            ).eval()
            self.trainable_features = nn.Sequential(bb.layer3, bb.layer4)
            self.features = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                          bb.layer1, bb.layer2, bb.layer3, bb.layer4)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, 1)

    def train(self, mode=True):
        super().train(mode)
        self.frozen_features.eval()
        return self

    @torch.inference_mode()
    def extract_frozen(self, x):
        return self.frozen_features(self.pool(x))

    def forward_trainable(self, z):
        return self.classifier(torch.flatten(self.avgpool(self.trainable_features(z)), 1))

    def forward(self, x):
        with torch.no_grad():
            z = self.frozen_features(self.pool(x))
        return self.forward_trainable(z)