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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=settings.figsize, dpi=settings.dpi)

    # RBF plot
    ax1.set_title("RBF fit")
    ax1.set_xlabel("x")
    ax1.set_ylim(-np.amax(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    # sine wave
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(2 * np.pi * x_sin)
    ax1.plot(
        x_sin, y_sin, color="gray", linestyle="--", zorder=0, label="True Sine Wave"
    )

    # RBF fit and noise points
    xs = np.linspace(0, 1, 100)
    xs = xs[:, np.newaxis]
    ax1.plot(
        xs, np.squeeze(model(jnp.asarray(xs))), "-", np.squeeze(data.x), data.y, "o"
    )

    # Basis function expansion
    ax2.set_title("RBF Kernels")
    ax2.set_xlabel("x")
    ax2.set_ylabel("$\phi(x)$")
    phi_values = model.expansion_module(jnp.asarray(xs))

    for i in range(model.m):
        ax2.plot(xs, phi_values[:, i], label=f"Kernel {i + 1}")

    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "fit.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
