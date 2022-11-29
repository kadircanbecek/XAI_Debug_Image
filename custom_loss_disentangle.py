import torch
from torch import nn
from torchvision.models import resnet18


class FeatureExtract(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        rn18 = resnet18(pretrained=pretrained)
        self.conv1 = rn18.conv1
        self.bn1 = rn18.bn1
        self.relu = rn18.relu
        self.maxpool = rn18.maxpool
        self.layer1 = rn18.layer1
        self.layer2 = rn18.layer2
        self.layer3 = rn18.layer3
        self.layer4 = rn18.layer4

        self.layer_nlm = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer_nlm(x)
        x = self.gap(x).reshape([-1, 32])

        return x


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear = nn.Linear(32, classes)

    def forward(self, x):
        x = self.linear(x)

        return x


class WholeModel(nn.Module):
    def __init__(self, classes, pretrained=True):
        super().__init__()
        self.fe = FeatureExtract(pretrained)
        self.cl = Classifier(classes)

    def forward(self, x):
        x = self.fe(x)
        x = self.cl(x)
        return x


def correlationloss(output):
    batch, dim = output.shape

    mean_of_batch = torch.mean(output)
    ones_vector = torch.ones((batch, dim))
    corr_mat_1 = output - mean_of_batch * ones_vector
    corr_mat_2 = torch.transpose(corr_mat_1, 0, 1)
    corr_mat = torch.matmul(corr_mat_2, corr_mat_1)
    loss = (1/(dim**2))*torch.sum(torch.abs(corr_mat))
    return loss


model_fe = FeatureExtract(pretrained=True)
inp = torch.rand([128, 3, 224, 224])
outp = model_fe(inp)

loss = correlationloss(outp)
print("done")
