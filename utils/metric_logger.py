from collections import deque, defaultdict
import torch
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=300):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.Tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.Tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def std_deviation(self):
        return np.std(self.series)

    @property
    def last_item(self):
        return self.series[-1]


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    @property
    def mean(self):
        return {name: meter.avg for name, meter in self.meters.items()}

    @property
    def median(self):
        return {name: meter.median for name, meter in self.meters.items()}

    @property
    def global_avg(self):
        return {name: meter.global_avg for name, meter in self.meters.items()}

    @property
    def std_deviation(self):
        return {name: meter.std_deviation for name, meter in self.meters.items()}

    @property
    def last_item(self):
        return {name: meter.last_item for name, meter in self.meters.items()}
