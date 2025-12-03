import torch
from torch import nn
from .Embedding import Embedding

class Decoder(nn.Module):
    def __init__(self, device, params):
        super(Decoder, self).__init__()
        self.device = device
        self.get_required_params(params)

        self.make_layers()

    def get_required_params(self, params):
        # Static embedding
        self.dim_in_embedding_lead_static = params['dim_in_embedding_lead_static']
        self.dim_out_embedding_lead_static = params['dim_out_embedding_lead_static']
        self.layers_embedding_lead_static = params['layers_embedding_lead_static']
        self.dropout_embedding_lead_static = params['dropout_embedding_lead_static']

        # Dynamic embedding
        self.dim_in_embedding_lead_dynamic = params['dim_in_embedding_lead_dynamic']
        self.dim_out_embedding_lead_dynamic = params['dim_out_embedding_lead_dynamic']
        self.layers_embedding_lead_dynamic = params['layers_embedding_lead_dynamic']
        self.dropout_embedding_lead_dynamic = params['dropout_embedding_lead_dynamic']

        # Transition
        self.dim_in_transition = params['dim_in_transition']
        self.dim_out_transition = params['dim_out_transition']
        self.layers_transition = params['layers_transition']
        self.dropout_transition = params['dropout_transition']

        # Decoder
        # self.dim_in_decoder_lead = self.dim_out_embedding_lead_dynamic + self.dim_out_embedding_lead_static + 4 + 1 # 4 for lead encodings, 1 for mask
        self.dim_in_decoder_lead = params['dim_in_decoder_lead']
        self.dim_h_decoder_lead = params['dim_h_decoder_lead']
        self.n_layers_decoder_lead = params['n_layers_decoder_lead']
        self.bidirectional_decoder_lead = params['bidirectional_decoder_lead']
        self.batch_first_decoder_lead = params['batch_first_decoder_lead']
        self.dim_out_decoder_lead = params['dim_out_decoder_lead']

    def make_layers(self):
        self.transition_h = Embedding(
            dim_in=self.dim_in_transition,
            dim_out=self.dim_out_transition,
            layer_sizes=self.layers_transition,
            dropout=self.dropout_transition
        )
        self.transition_c = Embedding(
            dim_in=self.dim_in_transition,
            dim_out=self.dim_out_transition,
            layer_sizes=self.layers_transition,
            dropout=self.dropout_transition
        )

        self.embedding_lead_static = Embedding(
            dim_in=self.dim_in_embedding_lead_static,
            dim_out=self.dim_out_embedding_lead_static,
            layer_sizes=self.layers_embedding_lead_static,
            dropout=self.dropout_embedding_lead_static
        )

        self.embedding_lead_dynamic = Embedding(
            dim_in=self.dim_in_embedding_lead_dynamic,
            dim_out=self.dim_out_embedding_lead_dynamic,
            layer_sizes=self.layers_embedding_lead_dynamic,
            dropout=self.dropout_embedding_lead_dynamic
        )

        self.decoder_lead = nn.LSTM(
            input_size=self.dim_in_decoder_lead, 
            hidden_size=self.dim_h_decoder_lead, 
            num_layers=self.n_layers_decoder_lead,
            bidirectional=self.bidirectional_decoder_lead,
            batch_first=self.batch_first_decoder_lead
        )
    
        self.rollout = nn.Linear(
            in_features=self.dim_h_decoder_lead,
            out_features=self.dim_out_decoder_lead,
        )

        self.rollout_swi = nn.Sequential(
            nn.Linear(in_features=self.dim_h_decoder_lead, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, sample, available, enc_h_lag, enc_c_lag):
        X_static = sample['X_static'].to(self.device) # (batch_size, dim_in_embedding_lead_static)
        X_dynamic = sample['X_dynamic_ERA5_lead'].to(self.device) # (lead, batch_size, dim_in_embedding_lead_dynamic + 4)

        lead = X_dynamic.shape[0]
        batch_size = X_dynamic.shape[1]

        # Embedding
        emb_dynamic = self.embedding_lead_dynamic(X_dynamic) # (lead, batch_size, dim_out_embedding_lead_dynamic)
        emb_static_lead = self.embedding_lead_static(X_static).unsqueeze(0).repeat(lead, 1, 1) # (lead, batch_size, dim_out_embedding_lead_static)
        emb_lead = torch.cat([emb_dynamic, emb_static_lead, X_dynamic[:, :, -4:]], dim=-1) # (lead, batch_size, dim_out_embedding_lead_dynamic + dim_out_embedding_lead_static + 4)

        # Transition
        h0 = self.transition_h(enc_h_lag) # (1, batch_size, dim_out_transition)
        c0 = self.transition_c(enc_c_lag) # (1, batch_size, dim_out_transition)

        # Decoder
        dec_h_lead, (_, _) = self.decoder_lead(emb_lead, (h0, c0)) # (lead, batch_size, dim_h_decoder_lead)

        # Rollout
        predictions = self.rollout(dec_h_lead)  # (lead, batch_size, dim_out_decoder_lead)
        predictions_swi = self.rollout_swi(dec_h_lead).squeeze(-1)

        return predictions, predictions_swi