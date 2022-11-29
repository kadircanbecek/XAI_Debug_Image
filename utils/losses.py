import torch.nn.functional as F
import torch

def BVAE_loss(x, x_hat, z_mean, z_logvar):
    """ Calculate Beta-VAE loss as in [1]
    Parameters
    ----------
    x : torch.tensor
        input data to the Beta-VAE
    x_hat : torch.tensor
        input data reconstructed by the Beta-VAE
    z_mean : torch.tensor
        mean of the latent distribution of shape
        (batch_size, latent_dim)
    z_logvar : torch.tensor
        diagonal log variance of the latent distribution of shape
        (batch_size, latent_dim)
    Returns
    -------
    loss : torch.tensor
        loss as a rank-0 tensor calculated as:
        reconstruction_loss + beta * KL_divergence_loss
    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
    # recon_loss = F.binary_cross_entropy(x_hat, x.detach(), reduction="mean")
    recon_loss = F.mse_loss(x_hat, x.detach(), reduction="mean")
    kl_loss = kl_div(z_mean, z_logvar)
    return recon_loss, kl_loss

def kl_div(mean, logvar):
    """Computes KL Divergence between a given normal distribution
    and a standard normal distribution
    Parameters
    ----------
    mean : torch.tensor
        mean of the normal distribution of shape (batch_size x latent_dim)
    logvar : torch.tensor
        diagonal log variance of the normal distribution of shape (batch_size x latent_dim)
    Returns
    -------
    loss : torch.tensor
        KL Divergence loss computed in a closed form solution
    """
    batch_loss = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)
    loss = batch_loss.sum()
    return loss

def correlationloss(output):
    output = output.cpu()
    batch, dim = output.shape

    mean_of_batch = torch.mean(output)
    ones_vector = torch.ones((batch, dim))
    corr_mat_1 = output - mean_of_batch * ones_vector
    corr_mat_2 = torch.transpose(corr_mat_1, 0, 1)
    corr_mat = torch.matmul(corr_mat_2, corr_mat_1)
    loss = (1 / (dim ** 2)) * torch.sum(torch.abs(corr_mat))
    return loss