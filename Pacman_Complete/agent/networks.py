"""Network definitions: NoisyNet layers and the dueling convolutional Q-network."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module):
    """Linear layer with factorised Gaussian noise on its parameters.

    Replaces epsilon-greedy exploration: the noise is part of the weights, so
    the amount of exploration is learned instead of being scheduled by hand.
    In eval mode only the mean parameters are used, which makes the policy
    deterministic.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0))
        self.bias_epsilon.copy_(epsilon_out)

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQNConvAux(nn.Module):
    """Dueling Q-network over a stack of maze images plus auxiliary features.

    The convolutional body reads the stacked 36x28 maze channels; the flattened
    output is concatenated with the auxiliary vector (score, lives, ghost modes
    and distances) and fed to two NoisyLinear heads estimating the state value
    V(s) and the action advantages A(s, a). They recombine into
    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).
    """

    def __init__(self, in_channels, aux_size, n_actions, H=36, W=28):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Run a dummy batch through the body to size the heads.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            flat_size = self.cnn(dummy).shape[1]
        self.value_stream = nn.Sequential(
            NoisyLinear(flat_size + aux_size, 512), nn.ReLU(), NoisyLinear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(flat_size + aux_size, 512), nn.ReLU(), NoisyLinear(512, n_actions)
        )

    def forward(self, img, aux):
        img = img.to(device)
        aux = aux.to(device)
        x = self.cnn(img)
        x = torch.cat([x, aux], dim=1)
        V = self.value_stream(x)
        A = self.adv_stream(x)
        return V + (A - A.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Draw a fresh noise sample in every NoisyLinear layer."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
