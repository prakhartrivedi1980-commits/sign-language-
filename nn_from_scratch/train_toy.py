
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .mlp import MLP, cross_entropy_loss
from .optimizers import Adam


# 1. Generate a two-class spiral dataset
def make_spirals(points_per_class=200, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)

    X = np.zeros((points_per_class * 2, 2))
    y = np.zeros(points_per_class * 2, dtype=int)

    for class_id in range(2):
        indices = slice(
            class_id * points_per_class,
            (class_id + 1) * points_per_class
        )

        radius = np.linspace(0, 1, points_per_class)
        theta = (
            np.linspace(0, 3 * np.pi, points_per_class)
            + class_id * np.pi
            + rng.normal(0, noise, points_per_class)
        )

        X[indices, 0] = radius * np.sin(theta)
        X[indices, 1] = radius * np.cos(theta)
        y[indices] = class_id

    return X, y


def main():
    X, y = make_spirals()

    # Convert class labels to one-hot vectors
    targets = np.eye(2)[y]

    # 2. Initialize our existing MLP
    model = MLP(
        layer_sizes=[2, 64, 64, 2],
        hidden_activation="tanh",
        output_activation="linear",
        seed=42
    )

    optimizer = Adam(learning_rate=0.01)

    epochs = 2000
    losses = []
    accuracies = []

    # 3. Training loop
    for epoch in range(epochs):
        logits = model.forward(X)

        loss, gradient = cross_entropy_loss(
            logits, targets
        )

        model.backward(gradient)
        optimizer.step(model)

        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y)

        losses.append(loss)
        accuracies.append(accuracy)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )

    # 4. Save training plots

    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent

# Create results folder inside project root
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to: {results_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")

    axes[1].plot(accuracies)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    output_path = results_dir / "toy_training.png"
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Plot saved successfully: {output_path}")
    print(f"File exists: {output_path.is_file()}")

    plt.close(fig)

    print("\nTraining completed!")
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Final training accuracy: {accuracies[-1]:.2%}")
    print("Plot saved to results/toy_training.png")


if __name__ == "__main__":
    main()
