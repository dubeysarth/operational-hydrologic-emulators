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
        self.dim_in_embedding_lag_dynamic_era5 = params['dim_in_embedding_lag_dynamic_era5']
        self.dim_in_embedding_lag_dynamic_gpm_final = params['dim_in_embedding_lag_dynamic_gpm_final']
        self.dim_in_embedding_lag_dynamic_gpm_late = params['dim_in_embedding_lag_dynamic_gpm_late']
        self.dim_out_embedding_lag_dynamic = params['dim_out_embedding_lag_dynamic']
        self.layers_embedding_lag_dynamic = params['layers_embedding_lag_dynamic']
        self.dropout_embedding_lag_dynamic = params['dropout_embedding_lag_dynamic']

        # Encoder
        # self.dim_in_encoder_lag = self.dim_out_embedding_lag_dynamic + self.dim_out_embedding_lag_static
        # self.dim_in_encoder_lag = self.dim_out_embedding_lag_dynamic * 3 + self.dim_out_embedding_lag_static + 4 + 3 # 4 for lag encodings, 3 for masks
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

        self.embedding_lag_dynamic_era5 = Embedding(
            dim_in=self.dim_in_embedding_lag_dynamic_era5,
            dim_out=self.dim_out_embedding_lag_dynamic,
            layer_sizes=self.layers_embedding_lag_dynamic,
            dropout=self.dropout_embedding_lag_dynamic
        )

        self.embedding_lag_dynamic_gpm_final = Embedding(
            dim_in=self.dim_in_embedding_lag_dynamic_gpm_final,
            dim_out=self.dim_out_embedding_lag_dynamic,
            layer_sizes=self.layers_embedding_lag_dynamic,
            dropout=self.dropout_embedding_lag_dynamic
        )

        self.embedding_lag_dynamic_gpm_late = Embedding(
            dim_in=self.dim_in_embedding_lag_dynamic_gpm_late,
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

        # X_dynamic_ERA5: torch.Size([365, 395, 24])
        # X_dynamic_GPM_Final: torch.Size([365, 395, 5])
        # X_dynamic_GPM_Late: torch.Size([365, 395, 5])
        # X_static: torch.Size([395, 77])
        # lag_encodings: torch.Size([365, 395, 4])
        # lag_era5: torch.Size([365])
        # lag_gpm_final: torch.Size([365])
        # lag_gpm_late: torch.Size([365])

        X_static = sample['X_static'].to(self.device)
        X_dynamic_ERA5 = sample['X_dynamic_ERA5'].to(self.device)
        X_dynamic_GPM_Final = sample['X_dynamic_GPM_Final'].to(self.device)
        X_dynamic_GPM_Late = sample['X_dynamic_GPM_Late'].to(self.device)
        lag_encodings = sample['lag_encodings'].to(self.device)
        
        mask_era5 = available['lag_era5'].to(self.device)
        mask_gpm_final = available['lag_gpm_final'].to(self.device)
        mask_gpm_late = available['lag_gpm_late'].to(self.device)

        lag = X_dynamic_ERA5.shape[0]
        batch_size = X_dynamic_ERA5.shape[1]

        mask_era5 = mask_era5.unsqueeze(-1).unsqueeze(-1) # (lag, 1, 1)
        mask_gpm_final = mask_gpm_final.unsqueeze(-1).unsqueeze(-1) # (lag, 1, 1)
        mask_gpm_late = mask_gpm_late.unsqueeze(-1).unsqueeze(-1) # (lag, 1, 1)

        # Embedding
        
        emb_dynamic_era5 = self.embedding_lag_dynamic_era5(X_dynamic_ERA5) * mask_era5 # (lag, batch_size, dim_out_embedding_lag_dynamic_era5)
        emb_dynamic_gpm_final = self.embedding_lag_dynamic_gpm_final(X_dynamic_GPM_Final) * mask_gpm_final # (lag, batch_size, dim_out_embedding_lag_dynamic_gpm_final)
        emb_dynamic_gpm_late = self.embedding_lag_dynamic_gpm_late(X_dynamic_GPM_Late) * mask_gpm_late # (lag, batch_size, dim_out_embedding_lag_dynamic_gpm_late)

        emb_static_lag = self.embedding_lag_static(X_static).unsqueeze(0).repeat(lag, 1, 1) # (batch_size, dim_out_embedding_lag_static)

        mask_era5 = mask_era5.repeat(1, batch_size, 1) # (lag, batch_size, 1)
        mask_gpm_final = mask_gpm_final.repeat(1, batch_size, 1) # (lag, batch_size, 1)
        mask_gpm_late = mask_gpm_late.repeat(1, batch_size, 1) # (lag, batch_size, 1)

        # Concatenate embeddings
        emb_lag = torch.cat([
            emb_dynamic_era5, 
            emb_dynamic_gpm_final, 
            emb_dynamic_gpm_late,
            emb_static_lag,
            mask_era5,
            mask_gpm_final,
            mask_gpm_late,
            lag_encodings
        ], dim=-1)

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