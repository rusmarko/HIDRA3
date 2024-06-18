import torch
from torch.utils.data import Dataset, DataLoader


class Train(Dataset):
    def __init__(self, config, data, valid_idx):
        super(Train, self).__init__()

        self.valid_tgs = data['valid tgs']
        self.geophy = data['geophy']
        self.tide = data['tide']
        self.ssh = data['ssh']
        self.tide_gauge_order = data['tide gauge order']
        self.valid_idx = valid_idx

        print(f'{len(self.valid_idx)} train instances.')

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        datai = self.valid_idx[i]

        geophy = self.geophy[datai - 72:datai + 72]

        ssh_in = torch.clone(self.ssh[:, datai - 72:datai])
        ssh_out = torch.clone(self.ssh[:, datai:datai + 72])
        tide = torch.clone(self.tide[:, datai - 72:datai + 72])

        valid_i = []
        for tgi, tg_name in enumerate(self.tide_gauge_order):
            if datai not in self.valid_tgs[tg_name]:
                ssh_in[tgi] = torch.nan
                tide[tgi] = torch.nan
                ssh_out[tgi] = torch.nan
            else:
                valid_i.append(tgi)
        if len(valid_i) > 2 and torch.rand(1).item() < .5:
            idx = torch.randperm(len(valid_i))
            start = int(torch.rand(1).item() * (len(valid_i) - 2)) + 2
            for i in range(start, len(valid_i)):
                tgi = valid_i[idx[i]]
                ssh_in[tgi] = torch.nan
                tide[tgi] = torch.nan

        return geophy, ssh_in, ssh_out, tide


class Data:
    def __init__(self, config):
        data = torch.load(f'{config.data_path}/train.pth')
        valid_tgs = data['valid tgs']
        valid_geophy = data['valid geophy']
        valid_tgs_union = set()
        for valid_tg in valid_tgs.values():
            valid_tgs_union.update(valid_tg)
        valid_idx = valid_geophy.intersection(valid_tgs_union)

        valid_train = set()
        valid_std_train = set()
        for idx in valid_idx:
            time = data['times'][idx]
            if not (config.test_range[0] <= time <= config.test_range[1]):
                valid_train.add(idx)
            elif time.year == 2020:
                valid_std_train.add(idx)
        valid_train = sorted(list(valid_train))
        valid_std_train = sorted(list(valid_std_train))

        self.train_lister = Train(config, data, valid_train)
        self.train = DataLoader(self.train_lister, batch_size=config.batch_size, num_workers=config.num_workers,
                                pin_memory=True, drop_last=False, shuffle=True)

        self.std_train_lister = Train(config, data, valid_std_train)
        self.std_train = DataLoader(self.std_train_lister, batch_size=config.batch_size, num_workers=config.num_workers,
                                    pin_memory=True, drop_last=False, shuffle=True)
