import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from .ode import VelocityField
from .hippocampus import HippocampalConstraint

class NGTFlow(nn.Module):
    def __init__(self, img_shape=(1, 96, 112, 96), hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.velocity_field = VelocityField(hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 3, padding=1)
        )
        self.symptom_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # MADRS prediction
        )
        self.hippo_constraint = HippocampalConstraint()

    def forward(self, x0, t, c=None):
        # x0: baseline MRI [B, 1, H, W, D]
        # t: time tensor
        z0 = self.encoder(x0)
        
        def dynamics(z, t):
            return self.velocity_field(z, t, c)
        
        z_t = odeint(dynamics, z0, t, method='dopri5')
        x_t = self.decoder(z_t[-1])
        madrs_pred = self.symptom_head(z_t[-1])
        
        return x_t, madrs_pred, z_t
