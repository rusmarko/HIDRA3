import torch
import yaml
from tqdm import tqdm

from hidra3.hidra3 import HIDRA

print('Loading data...')
test_data = torch.load('../data/test.pth', map_location='cpu')

with open('../data/normalization_parameters.yaml') as file:
    norm = yaml.safe_load(file)

print('Loading model...')
tide_gauge_locaitons = [
    'Koper',
    'Ancona',
    'Ortona',
    'Ravenna',
    'Tremiti',
    'Vieste',
    'Neretva',
    'Venice',
    'Sobra',
    'Stari Grad',
    'Vela Luka',
]
hidra = HIDRA(tide_gauge_locaitons)
try:
    hidra.load_state_dict(torch.load('../data/HIDRA3_parameters.pth', map_location='cpu'))
except RuntimeError as e:
    print('Loading of parameters failed, check that variable `tide_gauge_locations` has correct stations listed. Error:', e)
    exit()
hidra.eval()

print('Making predictions...')
predictions = {}
for location in tide_gauge_locaitons:
    predictions[location] = {}
for time in tqdm(test_data):
    instance = test_data[time]
    geophy = instance['geophy']
    sea_level = instance['sea level']

    ssh = torch.full((len(tide_gauge_locaitons), 72), torch.nan, dtype=torch.float32)
    tide = torch.full((len(tide_gauge_locaitons), 144), torch.nan, dtype=torch.float32)

    for location, data in sea_level.items():
        i = tide_gauge_locaitons.index(location)
        ssh[i] = data['ssh']
        tide[i] = data['tide']

    ssh = ssh[None].expand(geophy.shape[0], -1, -1).contiguous()
    tide = tide[None].expand(geophy.shape[0], -1, -1).contiguous()

    y, std = hidra(ssh, tide, geophy)
    for i, location in enumerate(tide_gauge_locaitons):
        y[:, i] = y[:, i] * norm['ssh'][location]['std']
        std[:, i] = std[:, i] * norm['ssh'][location]['std']
    pred = y.detach().cpu()
    pred_mean = torch.mean(pred, dim=0)
    std = std.detach().cpu()
    std = torch.sqrt(torch.mean(torch.square(std) + torch.square(pred), dim=0) - torch.square(pred_mean))
    pred = pred_mean

    for i, location in enumerate(tide_gauge_locaitons):
        predictions[location][time] = {
            'predicted': pred[i],
            'predicted std': std[i],
        }

torch.save(predictions, f'../data/predictions.pth')
print('Predictions saved to the data folder.')
