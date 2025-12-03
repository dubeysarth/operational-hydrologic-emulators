import torch
from torch import nn
from .Encoder import Encoder
from .Decoder import Decoder

class RainfallRunoff(nn.Module):
    def __init__(self, device, params):
        super(RainfallRunoff, self).__init__()
        self.device = device
        self.encoder = Encoder(device, params)
        self.decoder = Decoder(device, params)

    def forward(self, sample, available):
        enc_h_lag, enc_c_lag = self.encoder(sample, available)
        predictions, predictions_swi = self.decoder(sample, available, enc_h_lag, enc_c_lag)
        return {'Q': predictions, 'swi': predictions_swi}