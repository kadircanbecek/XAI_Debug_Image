import torch.nn.functional as F
import torch
from torch.autograd.functional import jacobian

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


def mnist_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data

    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design
    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # concept_dim is always 1
    concepts = concepts.squeeze(-1)
    aggregates = aggregates.squeeze(-1)

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.
        # print(grad_tensor.shape)
        # print(concepts.shape)
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x,
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # print(torch.cuda.memory_allocated(device="cuda"))
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_concepts
    J_hx = - torch.bmm(torch.cat(jacobians, dim=2), relevances)
    del jacobians
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated(device="cuda"))

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x,
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)
    del jacobians

    # Jacobian of concepts wrt x


    # bs x num_features x num_classes
    robustness_loss = J_yx + J_hx

    return robustness_loss.norm(p='fro')
