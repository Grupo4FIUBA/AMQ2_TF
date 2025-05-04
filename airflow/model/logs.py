import time

class LogEntry:
    
    def __init__(self, model, mse_lr, r2_lr, sqrt_mse_lr, config):
        self._timestamp = time.time()
        self.mse_lr = mse_lr
        self.name = model
        self.r2_lr = r2_lr
        self.sqrt_mse_lr =float(sqrt_mse_lr)
        self.config = config

    def to_dict(self):
        return {
            'timestamp': self._timestamp,
            'name' : self.name,
            'mse_lr': self.mse_lr,
            'r2_lr': self.r2_lr,
            'sqrt_mse_lr' : self.sqrt_mse_lr,
            'config' : self.config
        }