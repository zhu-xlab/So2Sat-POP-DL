import torchvision.models as models
import torch.nn as nn
import torch

from models.linresnet import linear_resnet50


class EOResNet(nn.Module):
    '''
    Earth Observation ResNet 50
    '''
    def __init__(self, input_channels, num_classes):
        super(EOResNet, self).__init__()

        ## set model features
        self.model = models.resnet50(pretrained=False) # pretrained=False just for debug reasons
        first_conv_layer = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        first_conv_layer = [first_conv_layer]

        first_conv_layer.extend(list(self.model.children())[1:-1])
        self.model= nn.Sequential(*first_conv_layer) 

        self.clf_layer = nn.Linear(in_features=2048, out_features = num_classes)
        self.clf_layer.apply(init_weights)        

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        outputs = self.clf_layer(outputs)
        return outputs


class EO2ResNet_OSM(nn.Module):
    '''
    Also using OSM Data
    '''
    def __init__(self, input_channels, num_classes, scale_factor=1):
        super(EO2ResNet_OSM, self).__init__()
        self.scale_factor = scale_factor
        num_features = [1024, 2048, 4096]
        self.cnn = EOResNet(input_channels, num_classes).model
        self.linear = linear_resnet50(scale_factor=scale_factor)
        if self.scale_factor != 1:
            self.lin_scale = nn.Sequential(
                nn.BatchNorm1d(num_features[1]*scale_factor),
                nn.Linear(num_features[1]*scale_factor, num_features[1]),
            )

        ## final depends on cnn output concat with linear output
        self.final = nn.Sequential(
            nn.Linear(in_features=num_features[2], out_features=num_features[1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=num_features[1], out_features=num_features[0]),
            nn.LeakyReLU(inplace=True)
            )
        self.clf_layer = nn.Linear(in_features=num_features[0],out_features=num_classes)
        self.final.apply(init_weights)
        self.clf_layer.apply(init_weights)

    def forward(self, inputs, osm_in):
        outputs = self.cnn(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        osm_in = osm_in.view(osm_in.shape[0],-1)
        osm_out = self.linear(osm_in)
        if self.scale_factor != 1:
            osm_out = self.lin_scale(osm_out)
        outputs = torch.cat((outputs, osm_out), dim=1)
        outputs = self.final(outputs)
        outputs = self.clf_layer(outputs)
        return outputs


def init_weights(layer, method = 'xavier normal'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == 'xavier normal':
            nn.init.xavier_normal_(layer.weight)
        elif method == 'kaiming normal':
            nn.init.kaiming_normal_(layer.weight)