from copy import deepcopy

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import LSTMDataset
from models import LSTMNetwork


class LSTMTrainer:
    def __init__(
        self,
        model: LSTMNetwork,
        optimizer: str | Optimizer = "adam",
        loss_fn: str | torch.nn.Module = "mse",
        learn_rate: float = 0.001,
        file_name: str = None,
    ) -> None:
        self.model = model
        self.learn_rate = learn_rate
        self.optimizer = self._select_optimizer(optimizer, learn_rate)
        self.loss_fn = self._select_loss_fn(loss_fn)
        self.file_name = file_name

    def train(
        self,
        train_dataset: LSTMDataset,
        valid_dataset: LSTMDataset,
        batch_size: int = 8,
        drop_last: bool = False,
        shuffle: bool = True,
        epochs: int = 10,
        quiet: bool = False,
    ) -> None:
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last or self.model.stateful
        self.shuffle = shuffle and not self.model.stateful  # stateful LSTM cannot shuffle data
        self.epochs = epochs
        self.train_history = []
        self.valid_history = []
        self.data_size = len(self.train_dataset)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2,
            drop_last=self.drop_last,  # wheb stateful, all batches should have the same size, drop the last batch to prevent batch size mis-match
            collate_fn=self.train_dataset.collate_fn,
        )

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=self.drop_last,  # wheb stateful, all batches should have the same size, drop the last batch to prevent batch size mis-match
            collate_fn=self.valid_dataset.collate_fn,
        )

        # set model to training mode
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # train
            self._train_one_epoch(train_dataloader=train_dataloader, quiet=quiet)
            # validate
            self._validate(valid_dataloader=valid_dataloader, quiet=quiet)

        self.model.eval()

        # save the model
        if self.file_name is not None:
            self.model.save_model(file_name=self.file_name)

        return deepcopy(self.train_history), deepcopy(self.valid_history)

    def _train_one_epoch(
        self, train_dataloader: DataLoader, quiet: bool = False
    ) -> None:

        # initialize hidden and cell state after each epoch
        hidden, cell = None, None

        # training loop
        for batch_idx, batch_sample in enumerate(train_dataloader):

            batch_instances = batch_sample.instance.to(self.model.device)
            batch_labels = batch_sample.label.to(self.model.device)

            # predict
            label_predict, hidden, cell = self.model(
                inputs=batch_instances,
                hidden_init=hidden,
                cell_init=cell
            )

            # calculate loss
            loss = self.loss_fn(label_predict, batch_labels)

            # record training loss history
            self.train_history.append(loss.detach().to("cpu").numpy())

            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # detach hidden and cell state for stateful LSTM
            hidden = hidden.detach() if self.model.stateful else None
            cell = cell.detach() if self.model.stateful else None

            if not quiet:
                self._print_train_info(loss, batch_idx + 1)

    def _validate(self, valid_dataloader: DataLoader, quiet: bool) -> float:
        total_loss = 0.0
        num_samples = 0

        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():

            # initialize hidden and cell state
            hidden, cell = None, None

            for batch_sample in valid_dataloader:
                batch_instances = batch_sample.instance.to(self.model.device)
                batch_labels = batch_sample.label.to(self.model.device)

                # predict
                label_predict, hidden, cell = self.model(
                    inputs=batch_instances,
                    hidden_init=hidden,
                    cell_init=cell
                )

                # calculate loss
                loss = self.loss_fn(label_predict, batch_labels)

                total_loss += loss.item() * batch_instances.shape[1]
                num_samples += batch_instances.shape[1]

        avg_loss = total_loss / num_samples

        if self.file_name is not None and avg_loss < min(self.valid_history, default=float("inf")):
            self.model.record_model()

        self.valid_history.append(avg_loss)

        self.model.train()  # reset model to training mode

        if not quiet:
            print(f"  validation loss: {avg_loss:>14.8f}\n")

        return avg_loss

    def _print_train_info(self, loss, batch_idx: int) -> None:

        last_batch = self.data_size // self.batch_size
        last_batch = last_batch if self.drop_last else last_batch + 1

        if batch_idx == last_batch:
            end = "\n"
        elif batch_idx % 50 == 0:
            end = "\r"
        else:
            return

        progress_percent = 100 * batch_idx * self.batch_size / self.data_size
        progress_percent = progress_percent if progress_percent < 100 else 100
        print(f"  loss: {loss:>14.8f}, [{progress_percent:>3.0f}%]", end=end)

    def _select_optimizer(
        self, optimizer: str | Optimizer, learn_rate: float
    ) -> Optimizer:
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                return torch.optim.Adam(self.model.parameters(), lr=learn_rate)
            elif optimizer.lower() == "sgd":
                return torch.optim.SGD(self.model.parameters(), lr=learn_rate)
            elif optimizer.lower() == "adamw":
                return torch.optim.AdamW(self.model.parameters(), lr=learn_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        elif isinstance(optimizer, Optimizer):
            return optimizer.add_param_group(self.model.parameters())
        else:
            raise TypeError(
                f"Optimizer should be a str or Optimizer object, a {type(optimizer)} was passed instead."
            )

    def _select_loss_fn(self, loss_fn: str | torch.nn.Module) -> torch.nn.Module:
        if isinstance(loss_fn, str):
            if loss_fn == "mse":
                return torch.nn.MSELoss(reduction="mean")
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
        elif isinstance(loss_fn, torch.nn.Module):
            return loss_fn
        else:
            raise TypeError(
                f"Loss function should be a str or nn.Module object, a {type(loss_fn)} was passed instead."
            )
