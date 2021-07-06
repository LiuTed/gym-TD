import random

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.ptr = 0
    
    def push(self, val):
        if self.ptr < self.capacity:
            self.buf.append(val)
        else:
            self.buf[self.ptr % self.capacity] = val
        self.ptr += 1
    
    def sample(self, num):
        return random.sample(self.buf, num)
    
    def __len__(self):
        return len(self.buf)
