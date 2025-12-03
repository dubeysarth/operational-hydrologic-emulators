import torch
from torch import nn
from .Embedding import Embedding

class Encoder(nn.Module):
    def __init__(self, device, params):
        super(Encoder, self).__init__()
        self.device = device
        self.get_required_params(params)

        self.make_layers()

    def get_required_params(self, params):
        # Static embedding
        self.dim_in_embedding_lag_static = params['dim_in_embedding_lag_static']
        self.dim_out_embedding_lag_static = params['dim_out_embedding_lag_static']
        self.layers_embedding_lag_static = params['layers_embedding_lag_static']
        self.dropout_embedding_lag_static = params['dropout_embedding_lag_static']

        # Dynamic embedding
        self.dim_in_embedding_lag_dynamic = params['dim_in_embedding_lag_dynamic']
        self.dim_out_embedding_lag_dynamic = params['dim_out_embedding_lag_dynamic']
        self.layers_embedding_lag_dynamic = params['layers_embedding_lag_dynamic']
        self.dropout_embedding_lag_dynamic = params['dropout_embedding_lag_dynamic']

        # Encoder
        # self.dim_in_encoder_lag = self.dim_out_embedding_lag_dynamic + self.dim_out_embedding_lag_static
        self.dim_in_encoder_lag = params['dim_in_encoder_lag']
        self.dim_h_encoder_lag = params['dim_h_encoder_lag']
        self.n_layers_encoder_lag = params['n_layers_encoder_lag']
        self.bidirectional_encoder_lag = params['bidirectional_encoder_lag']
        self.batch_first_encoder_lag = params['batch_first_encoder_lag']

    def make_layers(self):
        self.embedding_lag_static = Embedding(
            dim_in=self.dim_in_embedding_lag_static,
            dim_out=self.dim_out_embedding_lag_static,
            layer_sizes=self.layers_embedding_lag_static,
            dropout=self.dropout_embedding_lag_static
        )

        self.embedding_lag_dynamic = Embedding(
            dim_in=self.dim_in_embedding_lag_dynamic,
            dim_out=self.dim_out_embedding_lag_dynamic,
            layer_sizes=self.layers_embedding_lag_dynamic,
            dropout=self.dropout_embedding_lag_dynamic
        )

        self.encoder_lag = nn.LSTM(
            input_size=self.dim_in_encoder_lag, 
            hidden_size=self.dim_h_encoder_lag, 
            num_layers=self.n_layers_encoder_lag,
            bidirectional=self.bidirectional_encoder_lag,
            batch_first=self.batch_first_encoder_lag
        )

    def forward(self, sample, available):
        X_static = sample['X_static'].to(self.device) # (batch_size, dim_in_embedding_lag_static)
        X_dynamic = sample['X_dynamic_ERA5'].to(self.device) # (lag, batch_size, dim_in_embedding_lag_dynamic + 4)

        lag = X_dynamic.shape[0]
        batch_size = X_dynamic.shape[1]

        # Embedding
        emb_dynamic = self.embedding_lag_dynamic(X_dynamic) # (lag, batch_size, dim_out_embedding_lag_dynamic)
        emb_static_lag = self.embedding_lag_static(X_static).unsqueeze(0).repeat(lag, 1, 1) # (lag, batch_size, dim_out_embedding_lag_static)
        emb_lag = torch.cat([emb_dynamic, emb_static_lag, X_dynamic[:, :, -4:]], dim=-1) # (lag, batch_size, dim_out_embedding_lag_dynamic + dim_out_embedding_lag_static + 4)

        _, (enc_h_lag, enc_c_lag) = self.encoder_lag(emb_lag) # (n_layers_encoder_lag, batch_size, dim_h_encoder_lag)

        if self.bidirectional_encoder_lag:
            enc_h_lag_forward = enc_h_lag[-2, :, :]  # Forward direction
            enc_h_lag_backward = enc_h_lag[-1, :, :]  # Backward direction
            enc_h_lag = torch.cat((enc_h_lag_forward, enc_h_lag_backward), dim=-1) # (batch_size, 2 * dim_h_encoder_lag)

            enc_c_lag_forward = enc_c_lag[-2, :, :]  # Forward direction
            enc_c_lag_backward = enc_c_lag[-1, :, :]  # Backward direction
            enc_c_lag = torch.cat((enc_c_lag_forward, enc_c_lag_backward), dim=-1) # (batch_size, 2 * dim_h_encoder_lag)

        else:
            enc_h_lag = enc_h_lag[-1, :, :] # (batch_size, dim_h_encoder_lag)
            enc_c_lag = enc_c_lag[-1, :, :] # (batch_size, dim_h_encoder_lag)
        
        enc_h_lag = enc_h_lag.unsqueeze(0) # (1, batch_size, dim_h_encoder_lag or 2 * dim_h_encoder_lag)
        enc_c_lag = enc_c_lag.unsqueeze(0) # (1, batch_size, dim_h_encoder_lag or 2 * dim_h_encoder_lag)

        return enc_h_lag, enc_c_lag