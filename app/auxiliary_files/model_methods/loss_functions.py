import torch
import torch.nn as nn

from app.auxiliary_files.model_methods.dilate.dilate_loss import dilate_loss


def select_loss_function(config, device):
    return eval(config['name'])(config, device)


class Default:
    def __init__(self, config, device):
        # TODO refactor to do it generic
        super().__init__()
        self.criterion_name = config['criterion']
        criterion = config['criterion']
        if criterion == 'binary_cross_entropy':
            bce_loss = nn.BCELoss()
            sigmoid = nn.Sigmoid()
            self.criterion = lambda output, target: bce_loss(sigmoid(output), sigmoid(target))
        elif criterion == 'negative_log_likelihood':
            self.criterion = nn.NLLLoss()
        elif criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = lambda output, target: loss(output.float(), target.float())
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif criterion == 'exponential_mse':
            self.criterion = lambda input, target: torch.sum((input - target) ** config['rate']).mean()
        elif criterion == 'exponential_mse_sigmoid':
            self.criterion = lambda input, target: torch.sum(nn.Sigmoid()((input - target) ** config['rate'])).mean()
        elif criterion == 'weighted_mse':
            weights = torch.flip(torch.arange(config['prediction_interval']), [0])
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            weights = weights.float().to(device)
            self.criterion = lambda input, target: torch.sum(weights * (input - target) ** 2).mean()
        elif criterion == 'weighted_mse_sigmoid':
            weights = torch.flip(torch.arange(config['prediction_interval']), [0])
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            weights = weights.float().to(device)
            self.criterion = lambda input, target: torch.sum(nn.Sigmoid()(weights * (input - target) ** 2)).mean()
        else:
            raise Exception('Loss function criterion not recognized')

    def run(self, output, target):
        return self.criterion(output, target)


class VAE:
    def __init__(self, config, device):
        super().__init__()
        self.recon_criterion_name = config['recon_criterion']['name']
        if self.recon_criterion_name == 'dilate':
            self.alpha = config['recon_criterion']['alpha']
            self.gamma = config['recon_criterion']['gamma']
            self.device = device
        else:
            self.recon_criterion = eval('nn.' + self.recon_criterion_name)(**config['recon_criterion']['args'])
        self.kld_beta = config['kld_beta']

    def run(self, output, target):
        recon_x, mu, logvar, z = output
        if self.recon_criterion_name == 'dilate':
            recon_loss, _, _ = dilate_loss(target, recon_x, self.alpha, self.gamma, self.device)
        else:
            recon_loss = self.recon_criterion(recon_x, target)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld * self.kld_beta


class DILATE:
    def __init__(self, config, device):
        super().__init__()
        self.alpha = config['alpha'] if 'alpha' in config else 0.5
        self.gamma = config['gamma'] if 'gamma' in config else 0.1
        self.device = device

    def run(self, output, target):
        loss, loss_shape, loss_temporal = dilate_loss(target, output, self.alpha, self.gamma, self.device)
        return loss


class RSequenceVAE:
    def __init__(self, config, device):
        super().__init__()
        self.recon_criterion = eval('nn.' + config['recon_criterion']['name'])(**config['recon_criterion']['args'])
        self.beta_first_representation = config['beta_first_representation']
        self.beta_second_representation = config['beta_second_representation']
        self.beta_recurrent = 1 - self.beta_first_representation - self.beta_second_representation
        self.kld_first_representation = config['kld_beta_first_representation']
        self.kld_second_representation = config['kld_beta_second_representation']

    def run(self, output, target):
        (first_representation_outputs, first_representation_mu, first_representation_log_var, _), \
        (second_representation_outputs, second_representation_mu, second_representation_log_var, _), \
        un_padded_recurrent_outputs = output
        (first_representation_inputs, second_representation_inputs) = target

        kld_loss_first_representation = -0.5 * torch.sum(
            1 + first_representation_log_var - first_representation_mu.pow(2) - first_representation_log_var.exp())
        recon_loss_first_representation = self.recon_criterion(first_representation_outputs,
                                                               first_representation_inputs)
        loss_first_representation = recon_loss_first_representation + self.kld_first_representation * kld_loss_first_representation

        kld_loss_second_representation = -0.5 * torch.sum(
            1 + second_representation_log_var - second_representation_mu.pow(2) - second_representation_log_var.exp())
        recon_loss_second_representation = self.recon_criterion(second_representation_outputs,
                                                                second_representation_inputs)
        loss_second_representation = recon_loss_second_representation + self.kld_second_representation * kld_loss_second_representation

        recon_loss_recurrent = self.recon_criterion(un_padded_recurrent_outputs, second_representation_inputs)

        return self.beta_first_representation * loss_first_representation + self.beta_second_representation * loss_second_representation + self.beta_recurrent * recon_loss_recurrent
