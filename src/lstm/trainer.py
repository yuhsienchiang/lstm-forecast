from copy import deepcopy

import torch
from models import LSTMNetwork
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import LSTMDataset


class LSTMTrainer:
    """
    Trainer for training and validating an LSTM model.

    Attributes:
    -----------
    model: LSTMNetwork
        The LSTM model to be trained and validated.
    optimizer: str or Optimizer, optional (default="adam")
        The optimizer to use for training. It can be either a string representing the optimizer name
        ("adam", "sgd", "adamw") or an instance of torch.optim.Optimizer.
    loss_fn: str or torch.nn.Module, optional (default="mse")
        The loss function to use for training. It can be either a string representing the loss function
        name ("mse") or an instance of torch.nn.Module.
    learn_rate: float, optional (default=0.001)
        The learning rate for the optimizer.
    file_name: str, optional (default=None)
        The file name to save the trained model parameters.

    Methods:
    --------
    train(
        train_dataset: LSTMDataset,
        valid_dataset: LSTMDataset,
        *,
        batch_size: int = 8,
        drop_last: bool = False,
        shuffle: bool = True,
        epochs: int = 10,
        quiet: bool = False
    ) -> tuple[list[float], list[float]]:
        Train and validate the LSTM model.

    """
    def __init__(
        self,
        model: LSTMNetwork,
        *,
        optimizer: str | Optimizer = "adam",
        loss_fn: str | torch.nn.Module = "mse",
        learn_rate: float = 0.001,
        file_name: str = None,
    ) -> None:
        """
        Initialize the LSTMTrainer.

        Parameters:
        -----------
        model: LSTMNetwork
            The LSTM model to be trained and validated.
        optimizer: str or Optimizer, optional (default="adam")
            The optimizer to use for training. It can be either a string representing the optimizer name
            ("adam", "sgd", "adamw") or an instance of torch.optim.Optimizer. If passing in an Optimizer
            instance, please make sure the optimizer is initialized with the model parameters.
        loss_fn: str or torch.nn.Module, optional (default="mse")
            The loss function to use for training. It can be either a string representing the loss function
            name ("mse") or an instance of torch.nn.Module.
        learn_rate: float, optional (default=0.001)
            The learning rate for the optimizer.
        file_name: str, optional (default=None)
            The file name to save the trained model parameters. If None, the model will not be saved.

        """
        self.model = model
        self.learn_rate = learn_rate
        self.optimizer = self._select_optimizer(optimizer, learn_rate)
        self.loss_fn = self._select_loss_fn(loss_fn)
        self.file_name = file_name

    def train(
        self,
        train_dataset: LSTMDataset,
        valid_dataset: LSTMDataset,
        *,
        batch_size: int = 8,
        drop_last: bool = False,
        shuffle: bool = True,
        epochs: int = 10,
        quiet: bool = False,
    ) -> None:
        """
        Train and validate the LSTM model.

        Parameters:
        -----------
        train_dataset: LSTMDataset
            Dataset for training the model.
        valid_dataset: LSTMDataset
            Dataset for validating the model.
        batch_size: int, optional (default=8)
            Batch size for training.
            Batch size is overridden by window size when using stateful LSTM to ensure time series continuity.
        drop_last: bool, optional (default=False)
            Whether to drop the last incomplete batch during training.
            drop_last is set to True when using stateful LSTM to prevent batch size mis-match.
        shuffle: bool, optional (default=True)
            Whether to shuffle the training data before each epoch.
            Shuffling is disabled when using stateful LSTM.
        epochs: int, optional (default=10)
            Number of epochs for training.
        quiet: bool, optional (default=False)
            Whether to suppress training progress output.

        Returns:
        --------
        train_history: list[float]
            List containing training loss history.
        valid_history: list[float]
            List containing validation loss history.

        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size if not self.model.stateful else train_dataset.window_size # 
        self.drop_last = drop_last or self.model.stateful # when stateful, drop the last batch to prevent batch size mis-match
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
            drop_last=self.drop_last,
            collate_fn=self.train_dataset.collate_fn,
        )

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=self.drop_last,
            collate_fn=self.valid_dataset.collate_fn,
        )

        # set model to training mode
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # train
            self._train(train_dataloader=train_dataloader, quiet=quiet)
            # validate
            self._validate(valid_dataloader=valid_dataloader, quiet=quiet)

        self.model.eval()

        # save the model
        if self.file_name is not None:
            self.model.save_model(file_name=self.file_name)

        return deepcopy(self.train_history), deepcopy(self.valid_history)

    def _train(
        self, train_dataloader: DataLoader, quiet: bool = False
    ) -> None:
        """
        Perform one epoch of training loop.

        Parameters:
        -----------
        train_dataloader: DataLoader
            DataLoader for training data.
        quiet: bool, optional (default=False)
            Whether to suppress training progress output.

        """
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

            # set hidden and cell for next loop
            hidden = hidden.detach() if self.model.stateful else None
            cell = cell.detach() if self.model.stateful else None

            if not quiet:
                self._print_train_info(loss, batch_idx)

    def _validate(self, valid_dataloader: DataLoader, quiet: bool) -> float:
        """
        Perform model validation.

        Parameters:
        -----------
        valid_dataloader: DataLoader
            DataLoader for validation data.
        quiet: bool
            Whether to suppress validation progress output.

        Returns:
        --------
        avg_loss: float
            Average validation loss.

        """
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

        if self.file_name is not None and avg_loss <= min(self.valid_history, default=float("inf")):
            self.model.record_model()

        self.valid_history.append(avg_loss)

        self.model.train()  # reset model to training mode

        if not quiet:
            print(f"  validation loss: {avg_loss:>14.8f}\n")

        return avg_loss

    def _print_train_info(self, loss, batch_idx: int) -> None:
        """
        Print training information.

        Parameters:
        -----------
        loss: float
            Current training loss.
        batch_idx: int
            Index of the current batch. Index starts from 0.
        """
        batch_idx += 1
        last_batch = self.data_size // self.batch_size
        last_batch = (
            last_batch
            if self.drop_last or self.data_size % self.batch_size == 0
            else last_batch + 1
        )

        if batch_idx == last_batch:
            end = "\n"
        elif batch_idx % 10 == 0:
            end = "\r"
        else:
            return

        progress_percent = 100 * batch_idx * self.batch_size / self.data_size
        progress_percent = progress_percent if progress_percent < 100 else 100
        print(f"  loss: {loss:>14.8f}, [{progress_percent:>3.0f}%]", end=end)

    def _select_optimizer(
        self, optimizer: str | Optimizer, learn_rate: float
    ) -> Optimizer:
        """
        Select the optimizer.

        Parameters:
        -----------
        optimizer: str or Optimizer
            The optimizer to use for training. It can be either a string representing the optimizer name
            ("adam", "sgd", "adamw") or an instance of torch.optim.Optimizer.
        learn_rate: float
            The learning rate for the optimizer.

        Returns:
        --------
        Optimizer:
            The selected optimizer instance.

        """
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
            return optimizer
        else:
            raise TypeError(f"Optimizer should be a str or Optimizer object, a {type(optimizer)} was passed instead.")

    def _select_loss_fn(self, loss_fn: str | torch.nn.Module) -> torch.nn.Module:
        """
        Select the loss function.

        Parameters:
        -----------
        loss_fn: str or torch.nn.Module
            The loss function to use for training. It can be either a string representing the loss function
            name ("mse") or an instance of torch.nn.Module.

        Returns:
        --------
        torch.nn.Module:
            The selected loss function instance.

        """
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
