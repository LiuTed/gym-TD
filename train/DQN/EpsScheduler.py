class EpsScheduler(object):
    def __init__(self, eps, method, **kwargs):
        self.eps_init = eps
        self.eps = eps
        self.step = 0
        assert method in ['Linear', 'Exponential', 'Fixed'],\
            '''method must be in ['Linear', 'Exponential', 'Fixed']'''
        if method == 'Linear':
            self.method = method
            self.lower_bound = kwargs['lower_bound']
            self.target_steps = kwargs['target_steps']
            self._update = self._update_linear
        elif method == 'Exponential':
            self.method = method
            if kwargs.get('lower_bound') is not None:
                self.lower_bound = kwargs['lower_bound']
            else:
                self.lower_bound = 0.
            self.decay = kwargs['decay']
            self._update = self._update_exponential
        elif method == 'Fixed':
            self.method = method
            self._update = self._update_fixed
    
    def _update_linear(self):
        if self.step >= self.target_steps:
            self.eps = self.lower_bound
        else:
            self.eps = (self.step / self.target_steps) * (self.lower_bound - self.eps_init) + self.eps_init
    
    def _update_exponential(self):
        self.eps *= self.decay
        if self.eps < self.lower_bound:
            self.eps = self.lower_bound
    
    def _update_fixed(self):
        return
    
    def update(self):
        self._update()
        self.step += 1
        return self.eps
