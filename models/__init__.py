from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, mnist_cnn


def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    # TODO maybe here pass in the different initialization for the projectors?
    if model_cfg.name == "simsiam" or model_cfg.name == "simsiam_mnist":
        model = SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)

    elif model_cfg.name == "byol":
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == "simclr":
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == "swav":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
