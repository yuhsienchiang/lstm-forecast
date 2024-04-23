# LSTM Forecaster

The main focus of this repo is to provide an implementation of statful LSTM regressioner using PyTorch.

In TensorFlow, stateful LSTM can be implemented easily by setting ```stateful=True```. However, as a PyTorch user, I found limited resources, documentation, and implementation related to creating stateful LSTM using PyTorch.
As a reasult, after doing some research, I decided to create my own stateful LSTM using PyTorch and share my implementation.

## Usage

## Setup
1. Clone the project repository to your local machine
2. Move to the repository
    ```bash
    cd lstm-forecast
    ```
3. This project is managed using [Poetry](https://python-poetry.org/).  If Poetry isn't installed on your machine, please run the command bellow to install Poetry

    -  Linux, macOS, Windows (WSL)
        ```bash
        curl -sSL https://install.python-poetry.org | python3 -
        ```
    - Windows (Powershell)
        ```bash
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
        ```
    Checkout the [instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) on Poetry's documentation for further information.
4. Install the dependencies 
    ```bash
    poetry install
    ```
5. Train the model by executing ```train_runner.py```
    ```bash
    python train_runner.py
    ```

## Dataset

## References
- [Stateful LSTM in Keras](https://philipperemy.github.io/keras-stateful-lstm/)
- [Stateful LSTM on time series (pytorch)](https://www.kaggle.com/code/viliuspstininkas/stateful-lstm-on-time-series-pytorch)
