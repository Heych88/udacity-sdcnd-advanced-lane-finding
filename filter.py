import numpy as np
from collections import deque

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def moving_average(data):
    # calculates the moving average of the data
    # data : data to be averaged
    # return : the filtered average of the data
    return sum(data)/len(data)

class Filter():

    def __init__(self, max_length):
        self.queue = deque(maxlen=max_length)
        self.max_length = max_length

    def moving_average(self, data):
        # calculates the moving average of the filter as well as keeps track
        # of the time series  data
        # data : new data to be added to the queue to be filtered
        # return : the filtered average for the filter, -1 if error

        self.queue.appendleft(data)
        queue_length = len(self.queue)
        try:
            # find the moving average
            average = sum(self.queue) / queue_length
        except:
            average = -1

        if queue_length >= self.max_length:
            self.queue.pop()
        return average
