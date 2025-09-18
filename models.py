import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

def get_vgg16_model(pretrained: bool=False, custom_weights: str='') -> nn.Module:
    """
    Returns the VGG16 model
    :param pretrained: Loads the pretrained model if True
    :param custom_weights: If pretrained is False and custom_weights are given then they will be loaded into the model.
    :return: The VGG16 model
    """

    if pretrained:
        model = vgg16(weights=VGG16_Weights)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    else:
        model = vgg16(weights=None)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
        if custom_weights != '':
            model.load_state_dict(torch.load(custom_weights, weights_only=True))

    # print(model.classifier)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def get_custom_model():
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x):
            ...
            return x

    model = Network()

    return model