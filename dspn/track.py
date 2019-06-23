class Tracker:
    """ Track training statistics for different variables and view intermediate results through monitors. """

    def __init__(self, **monitors):
        self.monitors = monitors
        self.data = {k: [] for k in self.monitors.keys()}

    def new_epoch(self):
        for epochs in self.data.values():
            epochs.append([])
        for monitor in self.monitors.values():
            monitor.reset()

    def update(self, key, value):
        # store value in the current epoch
        self.data[key][-1].append(value)
        # notify the corresponding monitor about the update
        monitor = self.monitors[key]
        return monitor.update(value)


class Identity:
    def __init__(self):
        pass

    def update(self, value):
        return value

    def reset(self):
        pass


class Mean:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.n = 0

    def update(self, value):
        self.n += 1
        self.value += (value - self.value) / self.n
        return self.value


class ExpMean:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.value = 0
        self.debias = 1

    def update(self, value):
        m = self.momentum
        self.debias *= m
        self.value = m * self.value + (1 - m) * value
        return self.value / (1 - self.debias)
