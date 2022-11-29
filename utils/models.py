import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet18
import torch.nn.functional as F


class FeatureExtract(nn.Module):
    def __init__(self, features=32, pretrained=True):
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

        self.layer_nlm = nn.Conv2d(512, features, kernel_size=1, stride=1, padding=0, bias=False)

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


class TwoPartModel(nn.Module):
    def __init__(self, fe, cl):
        super().__init__()
        self.fe = fe
        self.cl = cl

    def forward(self, x):
        x = self.fe(x)
        x = self.cl(x)
        return x


class TwoPartModelVAE(nn.Module):
    def __init__(self, en, de, cl):
        super().__init__()
        self.en = en
        self.de = de
        self.cl = cl

    def forward(self, x):
        x = self.fe(x)
        x = self.cl(x)
        return x


class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32 * 20 * 20, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32 * 20 * 20)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


class ResNet_VAE(nn.Module):
    def __init__(self, resout=256, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256):
        super().__init__()
        self.resout = resout
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.resnet = FeatureExtract(resout)
        self.fc1 = nn.Linear(256, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def forward(self, x):
        mu, logvar, z, x_reconst = self.forward_train(x)
        if self.training:
            return mu, logvar, z, x_reconst
        return z

    def forward_train(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return mu, logvar, z, x_reconst

    def encode(self, x):
        x = self.resnet(x)  # ResNet

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')

    def reparametrization(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


class TwoPartResNetVAE(nn.Module):
    def __init__(self, resout=256, fc_hidden1=128, fc_hidden2=64, CNN_embed_dim=32, classes=10):
        super().__init__()
        self.resnetvae = ResNet_VAE(resout, fc_hidden1, fc_hidden2, CNN_embed_dim)
        self.cl = Classifier(classes)

    def forward(self, x):
        if self.training:
            mu, logvar, z, x_reconst = self.resnetvae(x)
            out = self.cl(z)
            return mu, logvar, out, x_reconst
        else:
            z = self.resnetvae(x)
            out = self.cl(z)
            return out
