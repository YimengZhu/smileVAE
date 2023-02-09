from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
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
            nn.Sigmoid
        )

    def forward(self, feature):
        latent = self.encoder(feature)
        reconstruct = self.decoder(latent)
        return reconstruct


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
    
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
            nn.Sigmoid
        )

    def forward(self, feature):
        latent = self.encoder(feature)
        mean, var = torch.chunk(latent, 2, dim=-1)

        std = var.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        latent_sample = mean + std * esp

        reconstruct= self.decoder(latent_sample)
        return reconstruct, mean, var