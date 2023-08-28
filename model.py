import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_property):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.property_predictor = nn.Linear(
            latent_dim, 
            num_property
        )


    def forward(self, feature):
        latent = self.encoder(feature)
        reconstruct = self.decoder(latent)
        properties = self.property_predictor(latent)
        return reconstruct, properties


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_property):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.property_predictor = nn.Linear(
            latent_dim, 
            num_property
        )


    def forward(self, feature):
        latent = self.encoder(feature)
        mean, var = torch.chunk(latent, 2, dim=-1)

        std = var.mul(0.5).exp_()
        esp = torch.randn(*mean.size())
        latent_sample = mean + std * esp

        reconstruct= self.decoder(latent_sample)

        properties = self.property_predictor(latent_sample)
        return reconstruct, properties[:,-1,:]


def make_model(model_name):
    if model_name == 'AE':
        return AutoEncoder(12, 32, 3, 3)
    if model_name == 'VAE':
        return VariationalAutoEncoder(12, 32, 3, 3)