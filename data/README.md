
### Structure of the Data

Data from [this link](https://doi.org/10.5281/zenodo.12571170) has the following structures:

- `train.pth`:
```
{
    'geophy': tensor of shape (M, 8, 9, 12), where M is number of time points, second dimension contains geophysical data, last two are spatial dimensions. Standardized.
    'ssh': tensor of shape (N, M), where N is number of locations. Standardized.
    'tide': tensor of shape (N, M), standardized.
    'times': list of time points in UTC, size M
    'valid geophy': set of indices of valid geophysical data (with no NANs and consequent times)
    'valid tgs': {
        location: set of indices of valid tide gauge data (with no NANs in past measurements, tide defined)
    },
    'tide gauge order': list of locations, as they appear in 'ssh' and 'tide'
}
```

- `test.pth`:
```
{
    prediction time in UTC: {
        'geophy': tensor of shape (50, 144, 8, 9, 12), where 50 is number of ensemble members, 144 is number of time points (72 before and 72 after the prediction point), third dimension contains geophysical data, last two are spatial dimensions. Standardized.
        'sea level': {
            location: {
                'ssh': tensor of shape 72, standardized,
                'tide': tensor of shape 144, standardized
            },
            ...
        }
    },
    ...
}
```

- `predictions.pth`:
```
[
    location: {
        prediction time in UTC: {
            'predicted': 72 values in cm, not standardized,
            'predicted std': 72 standard deviations in cm, not standardized,
        },
        ...
    },
    ...
]
```

- `tide gauges.pth`:
```
{
    location: {
        'name': name of the location,
        'thr low': threshold for low SSH values, not standardized
        'thr high': threshold for high SSH values, not standardized
        'eval times': {
            prediction times used for calculating metrics in the paper
        },
        'ground truth': {
            time in UTC: SSH value in cm, not standardized
            ...
        }
    },
    ...
}
```

### Adding New Locations

To add new locations in the Adriatic, you will need to change the following things:

1. Change `tide_gauge_locaitons` in `train.py` and `test.py` to include the new location.
2. Add data to `train.pth`: `ssh` and `tide` should be updated with new location data, as also `valid tgs` and `tide gauge order`.
3. Add data to `test.pth`: change `ssh` and `tide` for the new location.
4. Add data to `tide gauges.pth`: add the new location with its name, thresholds, evaluation times, and ground truth data.


