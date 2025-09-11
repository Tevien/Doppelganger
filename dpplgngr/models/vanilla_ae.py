import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl


class FC_Encoder(nn.Module):
    def __init__(self, input_size, mid_size, latent_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size)
        self.fc2 = nn.Linear(mid_size, latent_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2

class FC_Decoder(nn.Module):
    def __init__(self, latent_size, mid_size, output_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_size, mid_size)
        self.fc4 = nn.Linear(mid_size, output_size)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class VanillaLightning(pl.LightningModule):
    def __init__(self, input_size=8, mid_size=6, latent_size=3):
        super(VanillaLightning, self).__init__()
        self.encoder = FC_Encoder(input_size, mid_size, latent_size)
        self.decoder = FC_Decoder(latent_size, mid_size, input_size)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer