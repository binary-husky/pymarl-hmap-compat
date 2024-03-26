import torch
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch import nn, Tensor
# from abc import abstractmethod


class VanillaVAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 degenerate2ae = False,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.training = True

        # Part 1, encoder
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # Part 2, decoder
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, input_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.degenerate2ae = degenerate2ae

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return: (Tensor) List of latent codes
        """
        h_       = self.LeakyReLU(self.FC_input(input))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_) 

        return mean, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        h     = self.LeakyReLU(self.FC_hidden(z))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)

        return x_hat

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode(input)
        if not self.degenerate2ae:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return  [self.decode(z), input, mu, log_var]    #  x_hat, mean, log_var

    def loss_function_normal(self,
                      x, x_hat, mean, log_var, kld_loss_weight, recons_loss_weight,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = x_hat
        input = x
        mu = mean
        log_var = log_var

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        # 计算高斯分布和标准正态分布的KL散度
        if not self.degenerate2ae:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recons_loss*recons_loss_weight + kld_loss*kld_loss_weight
        else:
            kld_loss = torch.zeros_like(recons_loss)
            loss = recons_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def loss_function(self,
                      x, x_hat, mean, log_var, kld_loss_weight, recons_loss_weight, valid_mask=None
                      ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = x_hat[valid_mask]
        input = x[valid_mask]
        mu = mean[valid_mask]
        log_var = log_var[valid_mask]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        # 计算高斯分布和标准正态分布的KL散度
        if not self.degenerate2ae:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recons_loss*recons_loss_weight + kld_loss*kld_loss_weight
        else:
            kld_loss = torch.zeros_like(recons_loss)
            loss = recons_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    # Do I need this?
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    # Do I need this?
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]