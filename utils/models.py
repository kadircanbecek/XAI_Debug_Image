from abc import abstractmethod

import torch
from scipy import stats
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet18
import torch.nn.functional as F


class FeatureExtract(nn.Module):
    def __init__(self, features=32, pretrained=True):
        super().__init__()
        self.features = features
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
        x = self.gap(x).reshape([-1, self.features])

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
        self.fe = FeatureExtract(pretrained=pretrained)
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
    def __init__(self, resout=256, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256, image_size=32):
        super().__init__()
        self.resout = resout
        self.image_size = image_size
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
        return mu, logvar, z, x_reconst

    def forward_train(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
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
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear')
        return x

    def reparametrization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
        else:
            z = mu
        return z


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


class VaeConceptizer(nn.Module):
    """Variational Auto Encoder to generate basis concepts
    Concepts should be independently sensitive to single generative factors,
    which will lead to better interpretability and fulfill the "diversity"
    desiderata for basis concepts in a Self-Explaining Neural Network.
    VAE can be used to learn disentangled representations of the basis concepts
    by emphasizing the discovery of latent factors which are disentangled.
    """

    def __init__(self, image_size, channels, num_concepts, **kwargs):
        """Initialize Variational Auto Encoder
        Parameters
        ----------
        image_size : int
            size of the width or height of an image, assumes square image
        num_concepts : int
            number of basis concepts to learn in the latent distribution space
        """
        super().__init__()
        self.in_dim = image_size * image_size * channels
        self.z_dim = num_concepts
        self.encoder = VaeEncoder(self.in_dim, self.z_dim)
        self.decoder = VaeDecoder(self.in_dim, self.z_dim)

    def forward(self, x):
        """Forward pass through the encoding, sampling and decoding step
        Parameters
        ----------
        x : torch.tensor
            input of shape [batch_size x ... ], which will be flattened
        Returns
        -------
        concept_mean : torch.tensor
            mean of the latent distribution induced by the posterior input x
        x_reconstruct : torch.tensor
            reconstruction of the input in the same shape
        """
        concept_mean, concept_logvar = self.encoder(x)
        concept_sample = self.sample(concept_mean, concept_logvar)
        x_reconstruct = self.decoder(concept_sample)
        return (concept_mean.unsqueeze(-1),
                concept_logvar.unsqueeze(-1),
                x_reconstruct.view_as(x))

    def sample(self, mean, logvar):
        """Samples from the latent distribution using reparameterization trick
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is drawn from a standard normal distribution

        Parameters
        ----------
        mean : torch.tensor
            mean of the latent distribution of shape [batch_size x z_dim]
        log_var : torch.tensor
            diagonal log variance of the latent distribution of shape [batch_size x z_dim]

        Returns
        -------
        z : torch.tensor
            sample latent tensor of shape [batch_size x z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z


class VaeEncoder(nn.Module):
    """Encoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron
        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(100, z_dim)
        self.logvar_layer = nn.Linear(100, z_dim)

    def forward(self, x):
        """Forward pass of the encoder
        """
        x = self.FC(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar


class VaeDecoder(nn.Module):
    """Decoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron
        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim)
        )

    def forward(self, x):
        """Forward pass of a decoder"""
        x_reconstruct = torch.sigmoid(self.FC(x))
        return x_reconstruct


class ConvParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, in_channel, dropout=0.5, **kwargs):
        """Parameterizer for MNIST dataset.
        Consists of convolutional as well as fully connected modules.
        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        cl_sizes : iterable of int
            Indicates the number of kernels of each convolutional layer in the network. The first element corresponds to
            the number of input channels.
        kernel_size : int
            Indicates the size of the kernel window for the convolutional layers.
        hidden_sizes : iterable of int
            Indicates the size of each fully connected layer in the network. The first element corresponds to
            the number of input features. The last element must be equal to the number of concepts multiplied with the
            number of output classes.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.dropout = dropout

        cl_layers = []
        h = 8
        cl_layers.append(nn.Conv2d(in_channel, h, kernel_size=5))
        # TODO: maybe adaptable parameters for pool kernel size and stride
        cl_layers.append(nn.MaxPool2d(2, stride=2))
        cl_layers.append(nn.ReLU())
        while h < num_concepts * 4:
            cl_layers.append(nn.Conv2d(h, h * 32, kernel_size=5))
            # TODO: maybe adaptable parameters for pool kernel size and stride
            cl_layers.append(nn.MaxPool2d(2, stride=2))
            cl_layers.append(nn.ReLU())
            h = h * 32
        # dropout before maxpool
        cl_layers.insert(-2, nn.Dropout2d(self.dropout))

        cl_layers.append(nn.AdaptiveAvgPool2d(1))
        self.cl_layers = nn.Sequential(*cl_layers)

        fc_layers = []
        while h > num_concepts * num_classes * 4:
            fc_layers.append(nn.Linear(h, h // 4))
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.ReLU())
            h = h // 4
        fc_layers.append(nn.Linear(h, num_concepts * num_classes))
        fc_layers.append(nn.Dropout(self.dropout))
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass of MNIST parameterizer.
        Computes relevance parameters theta.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        return self.fc_layers(flattened).view(-1, self.num_concepts, self.num_classes)


class TwoPartResNetSENN(nn.Module):
    def __init__(self, resout=256, fc_hidden1=128, fc_hidden2=64, CNN_embed_dim=32, classes=10, in_channels=3):
        super().__init__()
        self.conceptizer = ResNet_VAE(resout, fc_hidden1, fc_hidden2, CNN_embed_dim)
        self.parameterizer = ConvParameterizer(CNN_embed_dim, classes, in_channels)
        self.aggregator = SumAggregator(classes)

    def forward(self, x):
        outputs = self.conceptizer(x)
        relevances = self.parameterizer(x)
        mu, logvar, concepts, x_reconst = outputs
        explanations = (concepts, (mu, logvar), relevances)
        predictions = self.aggregator(concepts, relevances)
        return predictions, explanations, x_reconst

    def traverse(self, matrix, dim, traversal_range, steps,
                 mean=None, std=None, use_cdf=True):
        """Linearly traverses through one dimension of a matrix independently

        Parameters
        ----------
        matrix: torch.tensor
            matrix whose dimensions will be traversed independently
        dim: int
            dimension of the matrix to be traversed
        traversal_range: float
            maximum value of the traversal range, if use_cdf is true this should be less than 0.5
        steps: int
            number of steps in the traversal range
        mean: float
            mean of the distribution for traversal using cdf
        std: float
            std of the distribution for traversal using cdf
        use_cdf: bool
            whether to use cdf traversal
        """

        if use_cdf:
            assert traversal_range < 0.5, \
                "If CDF is to be used, the traversal range must represent probability range of -0.5 < p < +0.5"
            assert mean is not None and std is not None, \
                "If CDF is to be used, mean and std has to be defined"
            prob_traversal = (1 - 2 * traversal_range) / 2  # from 0.45 to 0.05
            prob_traversal = stats.norm.ppf(prob_traversal, loc=mean, scale=std)[0]  # from 0.05 to -1.645
            traversal = torch.linspace(-1 * prob_traversal, prob_traversal, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        else:
            traversal = torch.linspace(-1 * traversal_range, traversal_range, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        return


class TwoPartConvSENN(nn.Module):
    def __init__(self, CNN_embed_dim=32, classes=10, in_channels=3, image_size=32):
        super().__init__()
        self.conceptizer = ConvConceptizer(image_size, CNN_embed_dim, 1, in_channels)
        self.parameterizer = ConvParameterizer(CNN_embed_dim, classes, in_channels)
        self.aggregator = SumAggregator(classes)

    def forward(self, x):
        outputs = self.conceptizer(x)
        relevances = self.parameterizer(x)
        mu, logvar, concepts, x_reconst = outputs
        explanations = (concepts, (mu, logvar), relevances)
        predictions = self.aggregator(concepts, relevances)
        return predictions, explanations, x_reconst

    def traverse(self, matrix, dim, traversal_range, steps,
                 mean=None, std=None, use_cdf=True):
        """Linearly traverses through one dimension of a matrix independently

        Parameters
        ----------
        matrix: torch.tensor
            matrix whose dimensions will be traversed independently
        dim: int
            dimension of the matrix to be traversed
        traversal_range: float
            maximum value of the traversal range, if use_cdf is true this should be less than 0.5
        steps: int
            number of steps in the traversal range
        mean: float
            mean of the distribution for traversal using cdf
        std: float
            std of the distribution for traversal using cdf
        use_cdf: bool
            whether to use cdf traversal
        """

        if use_cdf:
            assert traversal_range < 0.5, \
                "If CDF is to be used, the traversal range must represent probability range of -0.5 < p < +0.5"
            assert mean is not None and std is not None, \
                "If CDF is to be used, mean and std has to be defined"
            prob_traversal = (1 - 2 * traversal_range) / 2  # from 0.45 to 0.05
            prob_traversal = stats.norm.ppf(prob_traversal, loc=mean, scale=std)[0]  # from 0.05 to -1.645
            traversal = torch.linspace(-1 * prob_traversal, prob_traversal, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        else:
            traversal = torch.linspace(-1 * traversal_range, traversal_range, steps)
            matrix_traversal = matrix.clone()  # to avoid changing the matrix
            matrix_traversal[:, dim] = traversal
        return


class SumAggregator(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator.
        Aggregates concepts and relevances and returns the predictions for each class.
        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS, NUM_CLASSES).
        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class. Shape - (BATCH, NUM_CLASSES)

        """

        # concepts = concepts.unsqueeze(-1)
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)


class Conceptizer(nn.Module):
    def __init__(self):
        """
        A general Conceptizer meta-class. Children of the Conceptizer class
        should implement encode() and decode() functions.
        """
        super(Conceptizer, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

    def forward(self, x):
        """
        Forward pass of the general conceptizer.
        Computes concepts present in the input.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        Returns
        -------
        encoded : torch.Tensor
            Encoded concepts (batch_size, concept_number, concept_dimension)
        decoded : torch.Tensor
            Reconstructed input (batch_size, *)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

    @abstractmethod
    def encode(self, x):
        """
        Abstract encode function to be overridden.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        """
        pass

    @abstractmethod
    def decode(self, encoded):
        """
        Abstract decode function to be overridden.
        Parameters
        ----------
        encoded : torch.Tensor
            Latent representation of the data
        """
        pass


class ScalarMapping(nn.Module):
    def __init__(self, conv_block_size):
        """
        Module that maps each filter of a convolutional block to a scalar value
        Parameters
        ----------
        conv_block_size : tuple (int iterable)
            Specifies the size of the input convolutional block: (NUM_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH)
        """
        super().__init__()
        self.num_filters, self.filter_height, self.filter_width = conv_block_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_filters):
            self.layers.append(nn.Linear(self.filter_height * self.filter_width, 1))

    def forward(self, x):
        """
        Reduces a 3D convolutional block to a 1D vector by mapping each 2D filter to a scalar value.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH).
        Returns
        -------
        mapped : torch.Tensor
            Reduced input (BATCH, CHANNELS, 1)
        """
        x = x.view(-1, self.num_filters, self.filter_height * self.filter_width)
        mappings = []
        for f, layer in enumerate(self.layers):
            mappings.append(layer(x[:, [f], :]))
        return torch.cat(mappings, dim=1)


class ConvConceptizer(Conceptizer):
    def __init__(self, image_size, num_concepts, concept_dim, image_channels=1, encoder_channels=(10,),
                 decoder_channels=(16, 8), kernel_size_conv=5, kernel_size_upsample=(5, 5, 2),
                 stride_conv=1, stride_pool=2, stride_upsample=(2, 1, 2),
                 padding_conv=0, padding_upsample=(0, 0, 1), **kwargs):
        """
        CNN Autoencoder used to learn the concepts, present in an input image
        Parameters
        ----------
        image_size : int
            the width of the input image
        num_concepts : int
            the number of concepts
        concept_dim : int
            the dimension of each concept to be learned
        image_channels : int
            the number of channels of the input images
        encoder_channels : tuple[int]
            a list with the number of channels for the hidden convolutional layers
        decoder_channels : tuple[int]
            a list with the number of channels for the hidden upsampling layers
        kernel_size_conv : int, tuple[int]
            the size of the kernels to be used for convolution
        kernel_size_upsample : int, tuple[int]
            the size of the kernels to be used for upsampling
        stride_conv : int, tuple[int]
            the stride of the convolutional layers
        stride_pool : int, tuple[int]
            the stride of the pooling layers
        stride_upsample : int, tuple[int]
            the stride of the upsampling layers
        padding_conv : int, tuple[int]
            the padding to be used by the convolutional layers
        padding_upsample : int, tuple[int]
            the padding to be used by the upsampling layers
        """
        super(ConvConceptizer, self).__init__()
        self.num_concepts = num_concepts
        self.filter = False
        self.dout = image_size

        # Encoder params
        encoder_channels = (image_channels,) + encoder_channels
        kernel_size_conv = handle_integer_input(kernel_size_conv, len(encoder_channels))
        stride_conv = handle_integer_input(stride_conv, len(encoder_channels))
        stride_pool = handle_integer_input(stride_pool, len(encoder_channels))
        padding_conv = handle_integer_input(padding_conv, len(encoder_channels))
        encoder_channels += (num_concepts,)

        # Decoder params
        decoder_channels = (num_concepts,) + decoder_channels
        kernel_size_upsample = handle_integer_input(kernel_size_upsample, len(decoder_channels))
        stride_upsample = handle_integer_input(stride_upsample, len(decoder_channels))
        padding_upsample = handle_integer_input(padding_upsample, len(decoder_channels))
        decoder_channels += (image_channels,)

        # Encoder implementation
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(self.conv_block(in_channels=encoder_channels[i],
                                                out_channels=encoder_channels[i + 1],
                                                kernel_size=kernel_size_conv[i],
                                                stride_conv=stride_conv[i],
                                                stride_pool=stride_pool[i],
                                                padding=padding_conv[i]))
            self.dout = (self.dout - kernel_size_conv[i] + 2 * padding_conv[i] + stride_conv[i] * stride_pool[i]) // (
                    stride_conv[i] * stride_pool[i])

        self.encoder.append(Flatten())
        self.encoder_mean = nn.Linear(self.dout ** 2, concept_dim)
        self.encoder_logvar = nn.Linear(self.dout ** 2, concept_dim)

        # Decoder implementation
        self.unlinear = nn.Linear(concept_dim, self.dout ** 2)
        self.decoder = nn.ModuleList()
        decoder = []
        for i in range(len(decoder_channels) - 1):
            decoder.append(self.upsample_block(in_channels=decoder_channels[i],
                                               out_channels=decoder_channels[i + 1],
                                               kernel_size=kernel_size_upsample[i],
                                               stride_deconv=stride_upsample[i],
                                               padding=padding_upsample[i]))
            decoder.append(nn.ReLU(inplace=True))
        decoder.pop()
        decoder.append(nn.Tanh())
        self.decoder = nn.ModuleList(decoder)

    def sample(self, mean, logvar):
        """Samples from the latent distribution using reparameterization trick
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is drawn from a standard normal distribution

        Parameters
        ----------
        mean : torch.tensor
            mean of the latent distribution of shape [batch_size x z_dim]
        log_var : torch.tensor
            diagonal log variance of the latent distribution of shape [batch_size x z_dim]

        Returns
        -------
        z : torch.tensor
            sample latent tensor of shape [batch_size x z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z

    def encode(self, x):
        """
        The encoder part of the autoencoder which takes an Image as an input
        and learns its hidden representations (concepts)
        Parameters
        ----------
        x : Image (batch_size, channels, width, height)
        Returns
        -------
        encoded : torch.Tensor (batch_size, concept_number, concept_dimension)
            the concepts representing an image
        """
        encoded = x
        for module in self.encoder:
            encoded = module(encoded)
        encoded_mean = self.encoder_mean(encoded)
        encoded_logvar = self.encoder_logvar(encoded)

        encoded = self.sample(encoded_mean, encoded_logvar)

        return encoded_mean, encoded_logvar, encoded

    def forward(self, x):
        encoded_mean, encoded_logvar, encoded_out = self.encode(x)
        decoded = self.decode(encoded_out)
        return encoded_mean, encoded_logvar, encoded_out, decoded

    def decode(self, z):
        """
        The decoder part of the autoencoder which takes a hidden representation as an input
        and tries to reconstruct the original image
        Parameters
        ----------
        z : torch.Tensor (batch_size, channels, width, height)
            the concepts in an image
        Returns
        -------
        reconst : torch.Tensor (batch_size, channels, width, height)
            the reconstructed image
        """
        reconst = self.unlinear(z)
        reconst = reconst.view(-1, self.num_concepts, self.dout, self.dout)
        for module in self.decoder:
            reconst = module(reconst)
        return reconst

    def conv_block(self, in_channels, out_channels, kernel_size, stride_conv, stride_pool, padding):
        """
        A helper function that constructs a convolution block with pooling and activation
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_conv : int
            the stride of the deconvolution
        stride_pool : int
            the stride of the pooling layer
        padding : int
            the size of padding
        Returns
        -------
        sequence : nn.Sequence
            a sequence of convolutional, pooling and activation modules
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride_conv,
                      padding=padding),
            # nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=stride_pool,
                         padding=padding),
            nn.ReLU(inplace=True)
        )

    def upsample_block(self, in_channels, out_channels, kernel_size, stride_deconv, padding):
        """
        A helper function that constructs an upsampling block with activations
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_deconv : int
            the stride of the deconvolution
        padding : int
            the size of padding
        Returns
        -------
        sequence : nn.Sequence
            a sequence of deconvolutional and activation modules
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride_deconv,
                               padding=padding),
        )


class Flatten(nn.Module):
    def forward(self, x):
        """
        Flattens the inputs to only 3 dimensions, preserving the sizes of the 1st and 2nd.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (dim1, dim2, *).
        Returns
        -------
        flattened : torch.Tensor
            Flattened input (dim1, dim2, dim3)
        """
        return x.view(x.size(0), x.size(1), -1)


def handle_integer_input(input, desired_len):
    """
    Checks if the input is an integer or a list.
    If an integer, it is replicated the number of  desired times
    If a tuple, the tuple is returned as it is
    Parameters
    ----------
    input : int, tuple
        The input can be either a tuple of parameters or a single parameter to be replicated
    desired_len : int
        The length of the desired list
    Returns
    -------
    input : tuple[int]
        a tuple of parameters which has the proper length.
    """
    if type(input) is int:
        return (input,) * desired_len
    elif type(input) is tuple:
        if len(input) != desired_len:
            raise AssertionError("The sizes of the parameters for the CNN conceptizer do not match."
                                 f"Expected '{desired_len}', but got '{len(input)}'")
        else:
            return input
    else:
        raise TypeError(f"Wrong type of the parameters. Expected tuple or int but got '{type(input)}'")
