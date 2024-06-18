import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Training:
    def __init__(self, config, net, data):
        self.config = config
        self.net = net
        self.data = data

        self.lr = config.lr

        base_params = []
        std_params = []
        for name, param in self.net.named_parameters():
            if name.startswith('std'):
                std_params.append(param)
            else:
                base_params.append(param)

        self.optimizer = optim.AdamW(base_params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optimizer_std = optim.AdamW(std_params, lr=self.config.lr_std)

    def train(self):
        print('Training started.')
        self.net = self.net.to(self.config.device)
        self.net.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=self.config.lr / 100)

        for epoch in tqdm(range(self.config.epochs)):
            for data in self.data.train:
                geophy, ssh_in, ssh_out, tide = data
                geophy = geophy.to(self.config.device)
                ssh_in = ssh_in.to(self.config.device)
                ssh_out = ssh_out.to(self.config.device)
                tide = tide.to(self.config.device)

                y, std = self.net(ssh_in, tide, geophy)

                mask = ~torch.isnan(ssh_out)
                ssh_out[~mask] = 0
                loss = torch.mean(torch.square(ssh_out - y)[mask])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            scheduler.step()

        print('Training of Uncertainty Estimation module started.')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_std, T_max=self.config.epochs_std, eta_min=self.config.lr_std / 100)

        for epoch in tqdm(range(self.config.epochs_std)):
            for data in self.data.std_train:
                geophy, ssh_in, ssh_out, tide = data
                geophy = geophy.to(self.config.device)
                ssh_in = ssh_in.to(self.config.device)
                ssh_out = ssh_out.to(self.config.device)
                tide = tide.to(self.config.device)

                y, std = self.net(ssh_in, tide, geophy)

                mask = ~torch.isnan(ssh_out)
                means = ssh_out[mask].detach()
                stds = std[mask]
                observed = y[mask].detach()
                normal = torch.distributions.normal.Normal(means, stds)
                log_prob = -normal.log_prob(observed)
                loss = torch.mean(log_prob)

                self.optimizer_std.zero_grad()
                loss.backward()
                self.optimizer_std.step()

            scheduler.step()

        torch.save(self.net.state_dict(), f'{self.config.data_path}/HIDRA3_parameters.pth')
        print('Model weights saved to the data folder.')
