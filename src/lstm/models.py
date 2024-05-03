from copy import deepcopy

import torch
from torch import Tensor, nn


def set_device(device_type: str) -> torch.device:
    """
    Set the device for running PyTorch operations.

    Parameters:
    -----------
    device_type: str
        Type of device to use ("mps", "cuda", or any other value for CPU).

    Returns:
    --------
    device: torch.device
        Selected device for running PyTorch operations.
        When  "mps" or "cuda" is selected but not avaliable, the device will default to "cpu".

    """
    if device_type.lower() == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device_type.lower() == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    return device


class Attention(nn.Module):
    """
    Attention mechanism applied to input features.

    Attributes:
    -----------
    features_dim: int
        Dimensionality of the input features.
    device: torch.device
        Device for running PyTorch operations.
    fc: nn.Linear
        Linear layer to calculate attention scores.

    Methods:
    --------
    forward(inputs: Tensor) -> Tensor:
        Forward pass of the attention mechanism.

    """

    def __init__(self, features_dim: int, device: str) -> None:
        """
        Initialize the Attention module.

        Parameters:
        -----------
        features_dim: int
            Dimensionality of the input features.
        device: str
            Type of device to use ("mps", "cuda", or any other value for CPU).

        """
        super(Attention, self).__init__()
        self.features_dim = features_dim
        self.device = set_device(device)
        self.fc = nn.Linear(in_features=features_dim, out_features=1, device=self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the attention mechanism.

        Parameters:
        -----------
        inputs: Tensor, shape (batch_size, seq_len, features_dim)
            Input tensor.

        Returns:
        --------
        context: Tensor
            Context vector computed using attention mechanism.

        """
        # calculate weights
        a_c = self.fc(inputs)
        weights = torch.softmax(a_c, dim=-2, dtype=torch.float32)
        # return weighted sum
        context = torch.sum(weights * inputs, dim=-2, dtype=torch.float32)
        return context


class LSTMNetwork(nn.Module):
    """
    LSTM model for time series forecasting.

        Attributes:
        -----------
        input_dim: int
            Dimensionality of the input features.
        hidden_dim: int
            Dimensionality of the hidden state of the LSTM.
        fc_dim: int
            Dimensionality of the fully connected layer.
        output_dim: int
            Dimensionality of the output.
        attn_layer: bool
            Whether an attention layer is included.
        stateful: bool
            Whether stateful LSTM is used.
        device: torch.device
            Device the model is running on.

        Methods:
        --------
        forward(inputs: Tensor, hidden_init: Tensor = None, cell_init: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
            Forward pass of the LSTM network.
        record_model():
            Record the current model parameters as the best model.
        save_model(save_path: str = "lstm.pth"):
            Save the model parameters to a file.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        fc_dim: int = 16,
        output_dim: int = 1,
        attn_layer: bool = False,
        stateful: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the LSTMNetwork model.

        Parameters:
        -----------
        input_dim: int
            Dimensionality of the input features.
        hidden_dim: int, optional (default=64)
            Dimensionality of the hidden state of the LSTM.
        fc_dim: int, optional (default=16)
            Dimensionality of the fully connected layer.
        output_dim: int, optional (default=1)
            Dimensionality of the output.
        attn_layer: bool, optional (default=False)
            Whether to include an attention layer.
        stateful: bool, optional (default=False)
            Whether to use stateful LSTM, where hidden state and cell state are passed from one batch to the next.
        device: str, optional (default="cpu")
            Device to run the model on ("cpu" or "cuda").
        """
        super(LSTMNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.attn_layer = attn_layer
        self.stateful = stateful
        self.device = set_device(device)

        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim, device=self.device)

        if self.attn_layer:
            self.attn = Attention(features_dim=hidden_dim, device=device)

        self.output = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=fc_dim, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim, out_features=fc_dim, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim, out_features=fc_dim // 2, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim // 2, out_features=output_dim, device=self.device),
        )

    def forward(
        self, inputs: Tensor, hidden_init: Tensor = None, cell_init: Tensor = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the LSTM network.

        Parameters:
        -----------
        inputs: Tensor, shape (seq_len, batch_size, input_dim) or (seq_len, input_dim)
            Input time serise tensor.
        hidden_init: Tensor, optional
            Initial hidden state tensor.
        cell_init: Tensor, optional
            Initial cell state tensor.

        Returns:
        --------
        output: Tensor
            Output tensor of the network.
        hidden: Tensor
            Final hidden state tensor.
        cell: Tensor
            Final cell state tensor.

        """
        # only accept batched or unbatched 2D time serise inputs
        batch = inputs.shape[1] if len(inputs.shape) == 3 else None

        if self.attn_layer:
            hiddens = []

        # initialise hidden state and cell state
        hidden, cell = self._init_hidden(hidden_init, cell_init, batch)

        # pass through LSTM layer
        for input in inputs:
            hidden, cell = self.lstm_cell(input, (hidden, cell))

            # record hidden states for attention layer
            if self.attn_layer:
                hiddens.append(hidden)

        if self.attn_layer:
            # forward pass through attention layer and fully connected layer
            output = self.output(self.attn(torch.stack(hiddens, dim=-2)))
        else:
            # forward pass through fully connected layer
            output = self.output(hidden)

        return output, hidden, cell

    def _init_hidden(
        self, hidden: Tensor, cell: Tensor, batch: int
    ) -> tuple[Tensor, Tensor]:
        """
        Initialize hidden and cell states if not provided.

        Parameters:
        -----------
        hidden: Tensor, optional
            Initial hidden state tensor.
        cell: Tensor, optional
            Initial cell state tensor.
        batch: int
            Batch size.

        Returns:
        --------
        hidden: Tensor
            Initialized hidden state tensor.
        cell: Tensor
            Initialized cell state tensor.

        """

        if hidden is None or cell is None:
            shape = (batch, self.hidden_dim) if batch else (self.hidden_dim,)

            hidden = torch.zeros(*shape, device=self.device)
            cell = torch.zeros(*shape, self.hidden_dim, device=self.device)

        return hidden, cell

    def record_model(self):
        """
        Record the current model parameters as the best model.
        """
        self.best_model = deepcopy(self.state_dict())

    def save_model(self, save_path: str = "lstm.pth"):
        """
        Save the model parameters to a file.

        Parameters:
        -----------
        save_path: str, optional (default="lstm.pth")
            File path to save the model parameters.
        """
        target_state_dict = getattr(self, "best_model", self.state_dict())

        torch.save(target_state_dict, save_path)
