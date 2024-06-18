from math import sqrt

import pandas as pd
import torch


class Metrics:
    def __init__(self, tg):
        self.tg = tg
        self.ground_truth = self.tg['ground truth']

        self.acc_thr = 10
        self.thr_low = self.tg['thr low']
        self.thr_high = self.tg['thr high']
        self.window = 3
        self.tolerance = 10
        self.sensitivity = 0

        self.maes = []
        self.rmses = []
        self.bias = []
        self.ok = []

        # high SSH
        self.maes_high = []
        self.rmses_high = []
        self.bias_high = []
        self.ok_high = []

        # low SSH
        self.maes_low = []
        self.rmses_low = []
        self.bias_low = []
        self.ok_low = []

        # recall
        self.tp = self.fn = self.fp = 0
        self.count = set()

        self.tp_list = []
        self.fp_list = []
        self.fn_list = []

    def is_peak(self, time):
        window = 3

        if time not in self.ground_truth:
            return False
        sshi = self.ground_truth[time]
        for d in range(1, window + 1):
            time_temp = time - pd.to_timedelta(d, 'h')
            if time_temp not in self.ground_truth:
                if d == 1:
                    return False
                break
            if self.ground_truth[time_temp] >= sshi:
                return False
        for d in range(1, window + 1):
            time_temp = time + pd.to_timedelta(d, 'h')
            if time_temp not in self.ground_truth:
                if d == 1:
                    return False
                break
            if self.ground_truth[time_temp] > sshi:
                return False
        return True

    def append(self, l, items):
        items = items.view(-1)
        for i in range(len(items)):
            l.append(items[i].item())

    def add(self, gt, pred, time):
        gt = gt.clone()
        pred = pred.clone()
        mask = ~torch.isnan(gt)
        if torch.sum(mask) == 0:
            return

        eval_start = 0
        eval_end = 72

        self.append(self.maes, torch.abs(gt - pred)[mask])
        self.append(self.rmses, torch.square(gt - pred)[mask])
        self.append(self.bias, (pred - gt)[mask])
        self.append(self.ok, (torch.abs(gt - pred) <= self.acc_thr)[mask].to(torch.float64))

        # high SSH
        temp_mask = mask * (gt >= self.thr_high)
        self.append(self.maes_high, torch.abs(gt - pred)[temp_mask])
        self.append(self.rmses_high, torch.square(gt - pred)[temp_mask])
        self.append(self.bias_high, (pred - gt)[temp_mask])
        self.append(self.ok_high, (torch.abs(gt - pred) <= self.acc_thr)[temp_mask].to(torch.float64))

        # low SSH
        temp_mask = mask * (gt <= self.thr_low)
        self.append(self.maes_low, torch.abs(gt - pred)[temp_mask])
        self.append(self.rmses_low, torch.square(gt - pred)[temp_mask])
        self.append(self.bias_low, (pred - gt)[temp_mask])
        self.append(self.ok_low, (torch.abs(gt - pred) <= self.acc_thr)[temp_mask].to(torch.float64))

        # RECALL
        # peaks
        pred_peaks = {}
        pred_peaks_i = {}
        for i in range(eval_start + 1, eval_end - 1):
            time_temp = time + pd.to_timedelta(i, 'h')
            a = max(eval_start, i - self.window)
            b = min(eval_end, i + self.window + 1)
            if pred[i] == torch.max(pred[a:b]):
                pred_peaks[time_temp] = pred[i]
                pred_peaks_i[time_temp] = i
        gt_peaks = {}
        gt_peaks_i = {}
        for i in range(eval_start - 10, eval_end + 10):
            time_temp = time + pd.to_timedelta(i, 'h')
            if self.is_peak(time_temp):
                gt_peaks[time_temp] = self.ground_truth[time_temp]
                gt_peaks_i[time_temp] = i

        time_to_pred = {time + pd.to_timedelta(i, 'h'): pred[i] for i in range(eval_end)}
        time_to_gt = {time + pd.to_timedelta(i, 'h'): gt[i] for i in range(eval_end) if not torch.isnan(gt[i])}

        # TP, FN
        for peak in gt_peaks:
            if not (time + pd.to_timedelta(eval_start, 'h') <= peak < time + pd.to_timedelta(eval_end, 'h')) or \
                    gt_peaks[peak] < self.thr_high:
                continue
            self.count.add(peak)
            is_in_pred = False
            thr = min(gt_peaks[peak] - self.tolerance - self.sensitivity, self.thr_high)
            for i in range(-self.window, self.window + 1):
                time_temp = peak + pd.to_timedelta(i, 'h')
                if time_temp in time_to_pred and time_to_pred[time_temp] >= thr:
                    is_in_pred = True
                    break

            if is_in_pred:
                self.tp += 1
                self.tp_list.append({
                    'time': time,
                    'peak time': peak,
                    'i': gt_peaks_i[peak] + 1,
                })
            else:
                self.fn += 1
                self.fn_list.append({
                    'time': time,
                    'peak time': peak,
                    'i': gt_peaks_i[peak] + 1,
                })

        # FP
        for peak in pred_peaks:
            if pred_peaks[peak] + self.sensitivity < self.thr_high:
                continue
            ok = True
            for d in range(-4, 5):
                temp_time = peak + pd.to_timedelta(d, 'h')
                if temp_time not in self.ground_truth:
                    ok = False
                    break
            if not ok:
                continue
            is_in_gt = False
            thr = min(pred_peaks[peak] + self.sensitivity - self.tolerance, self.thr_high)
            for i in range(-self.window, self.window + 1):
                time_temp = peak + pd.to_timedelta(i, 'h')
                if time_temp in time_to_gt and time_to_gt[time_temp] >= thr:
                    is_in_gt = True
                    break
            if not is_in_gt:
                self.fp += 1
                self.fp_list.append({
                    'time': time,
                    'peak time': peak,
                    'i': pred_peaks_i[peak] + 1,
                })

    def calc_mean(self, t, sq=False):
        if len(t) == 0:
            return torch.nan
        t = torch.tensor(t)
        m = torch.mean(t).item()
        if sq:
            return sqrt(m)
        return m

    def get(self):
        mae = self.calc_mean(self.maes)
        rmse = self.calc_mean(self.rmses, sq=True)
        bias = self.calc_mean(self.bias)
        acc = self.calc_mean(self.ok) * 100
        mae_high = self.calc_mean(self.maes_high)
        rmse_high = self.calc_mean(self.rmses_high, sq=True)
        bias_high = self.calc_mean(self.bias_high)
        acc_high = self.calc_mean(self.ok_high) * 100
        mae_low = self.calc_mean(self.maes_low)
        rmse_low = self.calc_mean(self.rmses_low, sq=True)
        bias_low = self.calc_mean(self.bias_low)
        acc_low = self.calc_mean(self.ok_low) * 100

        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else torch.nan
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else torch.nan
        if recall == 0 or precision == 0:
            f1 = torch.nan
        else:
            f1 = 2 / (recall ** -1 + precision ** -1)

        return {
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'acc': acc,
            'mae high': mae_high,
            'rmse high': rmse_high,
            'bias high': bias_high,
            'acc high': acc_high,
            'mae low': mae_low,
            'rmse low': rmse_low,
            'bias low': bias_low,
            'acc low': acc_low,
            'recall': recall * 100,
            'precision': precision * 100,
            'f1': f1 * 100,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tp list': self.tp_list,
            'fp list': self.fp_list,
            'fn list': self.fn_list,
            'n': len(self.maes),
        }

    def to_str(self):
        metrics = self.get()
        output = (f'{self.tg["name"]}, n={metrics["n"]}\n'
                  f'  MAE: {metrics["mae"]:.2f}, RMSE: {metrics["rmse"]:.2f}, BIAS: {metrics["bias"]:.2f}, ACC: {metrics["acc"]:.2f} %\n'
                  f'  LOW MAE: {metrics["mae low"]:.2f}, RMSE: {metrics["rmse low"]:.2f}, BIAS: {metrics["bias low"]:.2f}, ACC: {metrics["acc low"]:.2f} %\n'
                  f'  HIGH MAE: {metrics["mae high"]:.2f}, RMSE: {metrics["rmse high"]:.2f}, BIAS: {metrics["bias high"]:.2f}, ACC: {metrics["acc high"]:.2f} %\n'
                  f'  RECALL: {metrics["recall"]:.2f}, PRECISION: {metrics["precision"]:.2f}, F1: {metrics["f1"]:.2f}\n\n')
        return output


predictions = torch.load(f'../data/predictions.pth')
tide_gauges = torch.load(f'../data/tide gauges.pth')

for name, tg in tide_gauges.items():
    if name not in predictions:
        continue
    if 'ground truth' not in tg:
        print('Ground truth not provided for', tg['name'])
        continue

    metric = Metrics(tg)

    for time in tg['eval times']:
        predicted = predictions[name][time]['predicted']
        gt = torch.full_like(predicted, torch.nan)
        for i in range(72):
            temp_time = time + pd.to_timedelta(i, 'h')
            if temp_time in tg['ground truth']:
                gt[i] = tg['ground truth'][temp_time]
        metric.add(gt, predicted, time)

    print(metric.to_str())
