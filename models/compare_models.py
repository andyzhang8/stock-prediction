import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)
import importlib.util

# Parameters
sequence_length = 250
ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2020-01-01"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_directory(directory, model_file, pth_file):
    """Dynamically load a model class and weights from a specified directory."""
    # Load the model class
    model_path = os.path.join(directory, model_file)
    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Inspect the checkpoint to determine input size
    checkpoint_path = os.path.join(directory, pth_file)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Dynamically get the input size from the checkpoint
    lstm_weight_shape = checkpoint["lstm.weight_ih_l0"].shape
    input_size = lstm_weight_shape[1]  # Second dimension of lstm weight

    # Initialize the model
    model = module.LSTMModel(
        input_size=input_size, hidden_size=200, num_layers=5, dropout=0.2
    ).to(device)

    # Load the model weights
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate a model and return metrics."""
    with torch.no_grad():
        predictions = model(x_test).squeeze().cpu().numpy()
        actual = y_test.cpu().numpy()

    # Calculate performance metrics
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    explained_var = explained_variance_score(actual, predictions)
    max_err = max_error(actual, predictions)

    # Return detailed metrics
    return mse, mae, rmse, r2, explained_var, max_err, predictions, actual


def load_data(data_loader_path, directory, dataset_subdir="dataset"):
    """Load the data loader dynamically and return test data."""
    spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Construct dataset path
    dataset_path = os.path.join(directory, dataset_subdir)

    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found in {dataset_path}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data_loader = module.StockDataLoader(
        ticker=ticker,
        sequence_length=sequence_length,
        data_dir=dataset_path,
        start_date=start_date,
        end_date=end_date,
    )
    x, y, scaler = data_loader.get_data()
    x_test = torch.tensor(x[int(0.7 * len(x)) :], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[int(0.7 * len(y)) :], dtype=torch.float32).to(device)

    return x_test, y_test


def plot_results(predictions_v4, actual, predictions_v5):
    """Plot and compare predictions from both models."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", color="blue", alpha=0.8)
    plt.plot(predictions_v4, label="v4 Predictions", color="orange", alpha=0.7)
    plt.plot(predictions_v5, label="v5 Predictions", color="green", alpha=0.7)
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("comparison_plot.png")
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_models.py <v4_dir> <v5_dir>")
        sys.exit(1)

    v4_dir = sys.argv[1]
    v5_dir = sys.argv[2]

    # File names
    model_file = "lstm_model.py"
    pth_file = "final_lstm_stock_prediction_model.pth"
    data_loader_file = "data_loader.py"

    # Load data using v4's data loader
    data_loader_path_v4 = os.path.join(v4_dir, data_loader_file)
    x_test, y_test = load_data(data_loader_path_v4, v4_dir)

    # Evaluate v4 model
    print("\nEvaluating v4 model...")
    model_v4 = load_model_from_directory(v4_dir, model_file, pth_file)
    (
        v4_mse,
        v4_mae,
        v4_rmse,
        v4_r2,
        v4_explained_var,
        v4_max_err,
        predictions_v4,
        actual,
    ) = evaluate_model(model_v4, x_test, y_test)

    print("\nv4 Model Metrics:")
    print(f"Mean Squared Error (MSE): {v4_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {v4_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {v4_rmse:.4f}")
    print(f"R-squared (R²): {v4_r2:.4f}")
    print(f"Explained Variance Score: {v4_explained_var:.4f}")
    print(f"Max Error: {v4_max_err:.4f}")

    # Evaluate v5 model
    print("\nEvaluating v5 model...")
    model_v5 = load_model_from_directory(v5_dir, model_file, pth_file)
    (
        v5_mse,
        v5_mae,
        v5_rmse,
        v5_r2,
        v5_explained_var,
        v5_max_err,
        predictions_v5,
        _,
    ) = evaluate_model(model_v5, x_test, y_test)

    print("\nv5 Model Metrics:")
    print(f"Mean Squared Error (MSE): {v5_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {v5_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {v5_rmse:.4f}")
    print(f"R-squared (R²): {v5_r2:.4f}")
    print(f"Explained Variance Score: {v5_explained_var:.4f}")
    print(f"Max Error: {v5_max_err:.4f}")

    # Compare results
    print("\nComparison of Models:")
    if v4_r2 > v5_r2:
        print("Better Model: v4")
    elif v4_r2 < v5_r2:
        print("Better Model: v5")
    else:
        print("Both models perform equally well.")

    # Plot comparison
    plot_results(predictions_v4, actual, predictions_v5)


if __name__ == "__main__":
    main()
