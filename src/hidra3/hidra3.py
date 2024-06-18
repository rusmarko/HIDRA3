import math
import torch
import torch.nn as nn


class HIDRA(nn.Module):
    def __init__(self, tide_gauge_locations):
        super().__init__()

        self.tide_gauge_locations = tide_gauge_locations
        self.no_tide_gauges = len(tide_gauge_locations)
        drop = .1

        # Geophysical Encoder
        self.wind = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2)), nn.ReLU(), nn.Dropout3d(drop),
            nn.Conv3d(64, 512, kernel_size=(2, 4, 5), stride=(2, 1, 1)),
        )

        self.pressure = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2)), nn.ReLU(), nn.Dropout3d(drop),
            nn.Conv3d(64, 512, kernel_size=(2, 4, 5), stride=(2, 1, 1)),
        )

        self.sst = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2)), nn.ReLU(), nn.Dropout3d(drop),
            nn.Conv3d(64, 64, kernel_size=(2, 4, 5), stride=(2, 1, 1)),
        )

        self.waves = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2)), nn.ReLU(), nn.Dropout3d(drop),
            nn.Conv3d(64, 64, kernel_size=(2, 4, 5), stride=(2, 1, 1)),
        )

        self.geophy_temporal = nn.Conv1d(512 * 2 + 64 * 2, 256, kernel_size=5)
        self.geophy = nn.ModuleList([
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout1d(drop)),
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout1d(drop)),
        ])

        # Feature Extraction
        self.enc_dim = 2 ** 9
        self.state_dim = 2 ** 13
        self.skip_dim = 2 ** 10

        self.mix = nn.ModuleList()
        for _ in range(self.no_tide_gauges):
            self.mix.append(nn.ModuleList([
                nn.Linear(72 + 144, self.enc_dim),
                nn.Linear(2 ** 13, self.enc_dim),
                nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * self.enc_dim, 2 * self.enc_dim), nn.SELU(), nn.Dropout(drop)),
                    nn.Sequential(nn.Linear(2 * self.enc_dim, 2 * self.enc_dim), nn.SELU(), nn.Dropout(drop)),
                    nn.Sequential(nn.Linear(2 * self.enc_dim, 2 * self.enc_dim), nn.SELU(), nn.Dropout(drop)),
                    nn.Sequential(nn.Linear(2 * self.enc_dim, 2 * self.enc_dim), nn.SELU(), nn.Dropout(drop)),
                ]),
                nn.Linear(2 * self.enc_dim, self.state_dim),
                nn.Linear(2 * self.enc_dim, self.state_dim),
                nn.Linear(2 * self.enc_dim, self.skip_dim),
            ]))

        # SSH Regression
        self.skip_invalid = nn.ModuleList()
        for _ in range(self.no_tide_gauges):
            self.skip_invalid.append(nn.Linear(self.state_dim, self.skip_dim))
        self.reg = nn.ModuleList()
        self.std = nn.ModuleList()
        for _ in range(self.no_tide_gauges):
            self.reg.append(nn.Linear(self.skip_dim + self.state_dim, 72))
            self.std.append(nn.Linear(self.skip_dim + self.state_dim, 72))

        # Xavier init
        layers = []
        for module in [self.wind, self.pressure, self.sst, self.waves, self.geophy_temporal, self.geophy, self.skip_invalid]:
            layers += list(module.modules())
        for mix in self.mix:
            layers += list(mix[2].modules())
            layers += list(mix[3].modules())
            layers += list(mix[4].modules())
            layers += list(mix[5].modules())
        for i in range(len(layers)):
            layer = layers[i]
            if not isinstance(layer, nn.Linear) and not isinstance(layer, nn.Conv3d) and not isinstance(layer, nn.Conv1d):
                continue
            next_layer = layers[i + 1] if i + 1 < len(layers) else None
            gain = 1
            if next_layer is not None and isinstance(next_layer, nn.SELU):
                gain = 3 / 4
            elif next_layer is not None and isinstance(next_layer, nn.ReLU):
                gain = math.sqrt(2)
            nn.init.xavier_normal_(layer.weight, gain=gain)
            nn.init.normal_(layer.bias, std=.1)

        # scaling weights
        with torch.no_grad():
            for mix in self.mix:
                for i, layer in enumerate(mix[2]):
                    layer[0].weight *= 1 / 2 ** i
                    layer[0].bias *= 1 / 2 ** i

            for i in range(self.no_tide_gauges):
                map = torch.cat((
                    torch.ones(self.skip_dim, dtype=torch.bool),
                    torch.zeros(self.state_dim, dtype=torch.bool),
                ), 0)
                self.reg[i].weight[:, map] *= (self.skip_dim + self.state_dim) / (2 * self.skip_dim)
                self.std[i].weight[:, map] *= (self.skip_dim + self.state_dim) / (2 * self.skip_dim)

                map = torch.cat((
                    torch.zeros(self.skip_dim, dtype=torch.bool),
                    torch.ones(self.state_dim, dtype=torch.bool),
                ), 0)
                self.reg[i].weight[:, map] *= (self.skip_dim + self.state_dim) / (2 * self.state_dim)
                self.std[i].weight[:, map] *= (self.skip_dim + self.state_dim) / (2 * self.state_dim)

    def forward(self, ssh, tide, geophy):
        """
        :param ssh: b 11 72
        :param tide: b 11 144
        :param geophy: b 144 8 9 12
        """

        ssh = ssh.clone()
        tide = tide.clone()
        geophy = geophy.clone()

        mask = (~torch.any(torch.isnan(ssh), dim=2)) & (~torch.any(torch.isnan(tide), dim=2))  # b 11
        invalid_i = (~mask).nonzero().T
        assert torch.all(torch.any(mask, dim=1))
        assert not torch.any(torch.all(ssh == 0, dim=2)) and not torch.any(torch.all(tide == 0, dim=2)), 'use nan instead of 0 to mark invalid values'

        torch.nan_to_num_(ssh)
        torch.nan_to_num_(tide)

        batch_size = geophy.shape[0]

        # Geophysical Encoder
        geophy = geophy.permute(0, 2, 1, 3, 4)  # b 8 144 9 12
        y0 = self.pressure(geophy[:, :1])  # b 512 36 1 1
        y1 = self.wind(geophy[:, 1:3])  # b 512 36 1 1
        y2 = self.sst(geophy[:, 3:4])  # b 64 36 1 1
        y3 = self.waves(geophy[:, 4:])  # b 64 36 1 1
        y = torch.cat((y0, y1, y2, y3), dim=1)  # b 1152 36 1 1
        x = y.squeeze(3).squeeze(3)  # b 1152 36

        x = self.geophy_temporal(x)  # b 256 32
        for layer in self.geophy:
            x = x + layer(x)
        atmos_features = x.view(batch_size, -1)  # b 8192

        # Feature Extraction
        state = torch.empty(batch_size, self.no_tide_gauges, self.state_dim, dtype=torch.float32, device=ssh.device)  # b 11 8192
        w = torch.empty(batch_size, self.no_tide_gauges, self.state_dim, dtype=torch.float32, device=ssh.device)  # b 11 8192
        skip_valid = torch.empty(batch_size, self.no_tide_gauges, self.skip_dim, dtype=torch.float32, device=ssh.device)  # b 11 1024
        for i in range(self.no_tide_gauges):
            enc, atm_red_lay, mixs, dec, to_w, skip_layer = self.mix[i]
            tg_enc = enc(torch.cat((
                ssh[:, i],
                tide[:, i]
            ), 1))
            atmos_reduced = atm_red_lay(atmos_features)
            x = torch.cat((
                tg_enc,
                atmos_reduced,
            ), 1)
            for mix in mixs:
                x = x + mix(x)
            state[:, i] = dec(x)
            w[:, i] = to_w(x)
            skip_valid[:, i] = skip_layer(x)

        # Feature Fusion
        state[invalid_i[0], invalid_i[1]] = 0
        w[invalid_i[0], invalid_i[1]] = -torch.inf
        w = torch.nn.functional.softmax(w, 1)
        state = (state * w).sum(1)

        # SSH Regression
        skip_invalid = torch.empty(batch_size, self.no_tide_gauges, self.skip_dim, dtype=torch.float32, device=state.device)  # b 11 1024
        for i in range(self.no_tide_gauges):
            skip_invalid[:, i] = self.skip_invalid[i](state)
        skip = skip_valid
        skip[invalid_i[0], invalid_i[1]] = skip_invalid[invalid_i[0], invalid_i[1]]

        y = torch.zeros(batch_size, self.no_tide_gauges, 72, dtype=torch.float32, device=state.device)
        std = torch.zeros(batch_size, self.no_tide_gauges, 72, dtype=torch.float32, device=state.device)
        for tgi, final in enumerate(self.reg):
            features = torch.cat((
                skip[:, tgi],
                state,
            ), 1)
            y[:, tgi] = final(features)

            std[:, tgi] = torch.nn.functional.softplus(self.std[tgi](features.detach()))

        return y, std
