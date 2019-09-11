import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from fspool import FSPool
from dspn import DSPN
import utils


def build_net(args):
    if args.dataset == "mnist":
        set_channels = 2
        set_size = 342
    elif args.dataset == "clevr-box":
        set_channels = 4
        set_size = 10
    elif args.dataset == "clevr-state":
        set_channels = 18
        set_size = 10

    use_convolution = args.dataset.startswith("clevr")
    hidden_dim = args.dim
    inner_lr = args.inner_lr
    iters = args.iters
    latent_dim = args.latent

    set_encoder_class = globals()[args.encoder]
    set_decoder_class = globals()[args.decoder]

    set_encoder = set_encoder_class(set_channels, latent_dim, hidden_dim)
    if set_decoder_class == DSPN:
        set_decoder = DSPN(
            set_encoder, set_channels, set_size, hidden_dim, iters, inner_lr
        )
    else:
        # the set_channels + 1 is for the additional mask feature that has to be predicted
        set_decoder = set_decoder_class(
            latent_dim, set_channels + 1, set_size, hidden_dim
        )

    if use_convolution:
        input_encoder = ConvEncoder(latent_dim)
    else:
        input_encoder = None

    net = Net(
        input_encoder=input_encoder, set_encoder=set_encoder, set_decoder=set_decoder
    )
    return net


class Net(nn.Module):
    def __init__(self, set_encoder, set_decoder, input_encoder=None):
        """
        In the auto-encoder setting, don't pass an input_encoder because the target set and mask is
        assumed to be the input.
        In the general prediction setting, must pass all three.
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.input_encoder = input_encoder
        self.set_decoder = set_decoder

        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, target_mask):
        if self.input_encoder is None:
            # auto-encoder, ignore input and use target set and mask as input instead
            latent_repr = self.set_encoder(target_set, target_mask)
            target_repr = latent_repr
        else:
            # set prediction, use proper input_encoder
            latent_repr = self.input_encoder(input)
            # note that target repr is only used for loss computation in training
            # during inference, knowledge about the target is not needed
            target_repr = self.set_encoder(target_set, target_mask)

        predicted_set = self.set_decoder(latent_repr)

        return predicted_set, (target_repr, latent_repr)


############
# Encoders #
############


class ConvEncoder(nn.Module):
    """ ResNet34-based image encoder to turn an image into a feature vector """

    def __init__(self, latent):
        super().__init__()
        resnet = torchvision.models.resnet34()
        self.layers = nn.Sequential(*list(resnet.children())[:-2])
        self.end = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            # now has 2x2 spatial size
            nn.Conv2d(512, latent, 2),
            # now has shape (n, latent, 1, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.end(x)
        return x.view(x.size(0), -1)


class FSEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + 1, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels, 1),
        )
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        return x


class FSEncoderSized(nn.Module):
    """ FSEncoder, but one feature in representation is forced to contain info about sum of masks """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels - 1, 1),
        )
        self.pool = FSPool(output_channels - 1, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x = x * mask  # mask invalid elements away
        x, _ = self.pool(x)
        # include mask information in representation
        x = torch.cat([x, mask.mean(dim=2) * 4], dim=1)
        return x


class RNFSEncoder(nn.Module):
    """ Relation Network with FSPool instead of sum pooling. """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channels + 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, output_channels, 1),
        )
        self.lin = nn.Linear(dim, output_channels)
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        # include mask in set features
        x = torch.cat([x, mask], dim=1)
        # create all pairs of elements
        x = torch.cat(utils.outer(x), dim=1)

        x = self.conv(x)

        # flatten pairs and scale appropriately
        n, c, l, _ = x.size()
        x = x.view(x.size(0), x.size(1), -1) / l / l

        x, _ = self.pool(x)
        return x


class RNSumEncoder(nn.Module):
    """ Relation Network with FSPool instead of sum pooling. """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channels + 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, output_channels, 1),
        )
        self.lin = nn.Linear(dim, output_channels)
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        # include mask in set features
        x = torch.cat([x, mask], dim=1)
        # create all pairs of elements
        x = torch.cat(utils.outer(x), dim=1)

        x = self.conv(x)

        # flatten pairs and scale appropriately
        n, c, l, _ = x.size()
        x = x.view(x.size(0), x.size(1), -1) / l / l

        x = x.sum(dim=2)
        return x


class RNMaxEncoder(nn.Module):
    """ Relation Network with FSPool instead of sum pooling. """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channels + 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, output_channels, 1),
        )
        self.lin = nn.Linear(dim, output_channels)
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        # include mask in set features
        x = torch.cat([x, mask], dim=1)
        # create all pairs of elements
        x = torch.cat(utils.outer(x), dim=1)

        x = self.conv(x)

        # flatten pairs and scale appropriately
        n, c, l, _ = x.size()
        x = x.view(x.size(0), x.size(1), -1) / l / l

        x, _ = x.max(dim=2)
        return x


############
# Decoders #
############


class MLPDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, output_channels * set_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.view(x.size(0), self.output_channels, self.set_size)
        # assume last channel predicts mask
        features = x[:, :-1]
        mask = x[:, -1]
        # match output signature of DSPN
        return [features], [mask], None, None


class RNNDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.dim = dim
        self.lin = nn.Linear(input_channels, dim)
        self.model = nn.LSTM(1, dim, 1)
        self.out = nn.Conv1d(dim, output_channels, 1)

    def forward(self, x):
        # use input feature vector as initial cell state for the LSTM
        cell = x.view(x.size(0), -1)
        cell = self.lin(cell)
        # zero input of size set_size to get set_size number of outputs
        dummy_input = torch.zeros(self.set_size, cell.size(0), 1, device=cell.device)
        # initial hidden state of zeros
        dummy_hidden = torch.zeros(1, cell.size(0), self.dim, device=cell.device)
        # run the LSTM
        cell = cell.unsqueeze(0)
        output, _ = self.model(dummy_input, (dummy_hidden, cell))
        # project into correct number of output dims
        output = output.permute(1, 2, 0)
        output = self.out(output)
        # assume last channel predicts mask
        features = output[:, :-1]
        mask = output[:, -1]
        # match output signature of DSPN
        return [features], [mask], None, None
