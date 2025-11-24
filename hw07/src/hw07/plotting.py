import jax.numpy as jnp
import random
from flax import nnx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import structlog
from sklearn.inspection import DecisionBoundaryDisplay

from .config import PlottingSettings
from .data import Data
from .model import MLP

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_boundry(
    model: MLP,
    data: Data,
    settings: PlottingSettings,
    name: str = "Decision_",
):
    feature_1, feature_2 = np.meshgrid(
        np.linspace(-15, 15),
        np.linspace(-15, 15),
    )
    grid = jnp.asarray(
        np.vstack([feature_1.ravel(), feature_2.ravel()]).T,
    )
    y = np.reshape(
        nnx.sigmoid(model(grid)),
        feature_1.shape,
    )
    display = DecisionBoundaryDisplay(
        xx0=feature_1,
        xx1=feature_2,
        response=y,
    )
    display.plot(alpha=0.5, cmap="viridis")
    display.ax_.scatter(
        data.x[:, 0],
        data.x[:, 1],
        c=data.y,
        edgecolor="black",
        cmap="viridis",
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"{name}boundry.pdf"
    output_path = settings.output_dir / pdf_name
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_feature_activation(
    model: MLP,
    data: Data,
    key: nnx.Rngs,
    settings: PlottingSettings,
    grid_resolution: int = 100,
    n_plots: int = 30,
):
    GRID_MIN = -15.0
    GRID_MAX = 15.0

    feature_1, feature_2 = np.meshgrid(
        np.linspace(GRID_MIN, GRID_MAX, grid_resolution),
        np.linspace(GRID_MIN, GRID_MAX, grid_resolution),
    )

    grid_inputs = jnp.asarray(
        np.vstack([feature_1.ravel(), feature_2.ravel()]).T,
    )

    x = model(grid_inputs, get_logits=True)
    all_activations = model.ae(x, get_acts=True)
    _, d = all_activations.shape
    for i in range(n_plots):
        feature_idx = random.randint(0, d)
        heatmap_data = np.reshape(
            np.asarray(all_activations[:, feature_idx]),
            feature_1.shape,
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        c = ax.imshow(
            heatmap_data,
            extent=[GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX],
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )

        fig.colorbar(c, ax=ax, label=f"Activation Magnitude of Feature {feature_idx}")

        ax.scatter(
            data.x[:, 0],
            data.x[:, 1],
            c=data.y,
            edgecolor="black",
            cmap="gray",
            s=10,
            alpha=0.5,
        )
        ax.set_title(f"Feature {feature_idx} Receptive Field")
        ax.set_xlabel("Input Dimension 1 (X)")
        ax.set_ylabel("Input Dimension 2 (Y)")

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.output_dir / f"feature_{feature_idx}_heatmap.pdf"
        plt.savefig(output_path)
        plt.close(fig)
