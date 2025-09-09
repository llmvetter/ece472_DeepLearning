import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import NNXLinearModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXLinearModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the RBF fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("RBF fit")
    ax.set_xlabel("x")
    ax.set_ylim(-np.amax(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)
    x_sin = np.linspace(0, 2, 500)
    y_sin = np.sin(2 * np.pi * x_sin)
    ax.plot(
        x_sin, y_sin, color="gray", linestyle="--", zorder=0, label="True Sine Wave"
    )

    xs = np.linspace(0, 2, 100)
    xs = xs[:, np.newaxis]
    ax.plot(
        xs, np.squeeze(model(jnp.asarray(xs))), "-", np.squeeze(data.x), data.y, "o"
    )

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "fit.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
