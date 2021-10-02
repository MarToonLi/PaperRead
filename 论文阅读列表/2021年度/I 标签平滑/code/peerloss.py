class PeerBinaryClassifier(BinaryClassifier):
    def __init__(self, model, learning_rate, loss_func, alpha=1.):
        super(PeerBinaryClassifier, self).__init__(model, learning_rate, loss_func)
        self.alpha = alpha

    def train(self, X, y, X_, y_):
        self.model.train()

        y_pred = self.model(X)
        if self.ac_fn:
            y_pred = self.ac_fn(y_pred)
        y = torch.tensor(y, dtype=torch.float)

        y_pred_ = self.model(X_)
        if self.ac_fn:
            y_pred_ = self.ac_fn(y_pred_)
        y_ = torch.tensor(y_, dtype=torch.float)

        loss = self.loss(y_pred, y) - self.alpha * self.loss(y_pred_, y_)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def fit(self, X_train, y_train, X_val=None, y_val=None, episodes=100, batchsize=None, batchsize_=None,
            val_interval=20, log_interval=100, logger=None):
        if self.transform_y:
            y_train[y_train == 0] = -1
            if y_val is not None:
                y_val[y_val == 0] = -1

        losses, train_acc, val_acc = [], [], []
        batchsize = batchsize or len(X_train)
        batchsize_ = batchsize_ or len(X_train)
        m = X_train.shape[0]

        for ep in range(episodes):
            mb_idxes = np.random.choice(m, batchsize, replace=False)
            mb_X_train, mb_y_train = X_train[mb_idxes], y_train[mb_idxes]
            mb_X_train_ = X_train[np.random.choice(m, batchsize_, replace=False)]
            mb_y_train_ = y_train[np.random.choice(m, batchsize_, replace=False)]
            loss = self.train(mb_X_train, mb_y_train, mb_X_train_, mb_y_train_)
            losses.append(loss)
            
            if ep % val_interval == 0 and X_val is not None and y_val is not None:
                train_acc.append(self.val(X_train, y_train))
                val_acc.append(self.val(X_val, y_val))
            if logger is not None and ep % log_interval == 0:
                logger.record_tabular('ep', ep)
                logger.record_tabular('loss', np.mean(losses[-log_interval:]))
                logger.record_tabular('train_acc', np.mean(train_acc[-log_interval//val_interval:]))
                if X_val is not None and y_val is not None:
                    logger.record_tabular('val_acc', np.mean(val_acc[-log_interval//val_interval:]))
                logger.dump_tabular()

        return {
            'loss': losses,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
