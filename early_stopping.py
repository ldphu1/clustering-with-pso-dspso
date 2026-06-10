class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, verbose=True):
        """
        :param patience: Số lượng epoch tối đa chịu đựng việc không cải thiện trước khi dừng.
        :param min_delta: Mức giảm tối thiểu để được coi là "có cải thiện".
        :param verbose: Nếu True, in ra thông báo khi kích hoạt dừng sớm.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss, epoch):
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n[!] Kích hoạt Early Stopping tại Epoch {epoch}.")
                    print(f"    Lý do: Fitness không giảm quá {self.min_delta} trong {self.patience} epoch liên tiếp.")

        return self.early_stop