from torch import nn
from torchvision import models
from nest import register


@register
def ft_resnet(mode: str = 'resnet50', fc_or_fcn: str = 'fc', num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Finetune resnet.
    """

    class FT_Resnet(nn.Module):
        def __init__(self, mode='resnet50', fc_or_fcn='fc', num_classes=10, pretrained=True):
            super(FT_Resnet, self).__init__()
            
            if mode=='resnet50':
                model = models.resnet50(pretrained=pretrained)
            elif mode=='resnet101':
                model = models.resnet101(pretrained=pretrained)
            elif mode=='resnet152':
                model = models.resnet152(pretrained=pretrained)
            else:
                model = models.resnet18(pretrained=pretrained)

            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
            self.num_classes = num_classes
            self.num_features = model.layer4[1].conv1.in_channels
            self.fc_or_fcn = fc_or_fcn
            if self.fc_or_fcn=='fc':
            	self.classifier = nn.Linear(self.num_features, num_classes)
            else:
            	self.classifier = nn.Conv2d(self.num_features, self.num_classes, 1, 1)
            self.avg = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self.features(x)
            if self.fc_or_fcn=='fc':
            	x = self.avg(x).view(-1, self.num_features)
            	x = self.classifier(x)
            else:
            	x = self.classifier(x)            	
            	x = self.avg(x).view(-1, self.num_classes)
            return x

    return FT_Resnet(mode,  fc_or_fcn, num_classes, pretrained)
