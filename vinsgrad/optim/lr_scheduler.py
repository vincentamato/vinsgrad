class LRScheduler:
    def __init__(self, optimizer, patience=5, factor=0.1, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.patience:
            self.bad_epochs = 0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
            print(f"Reducing learning rate to {param_group['lr']}")