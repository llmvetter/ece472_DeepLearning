import jax.numpy as jnp
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


def plot_data(data: Data, settings: PlottingSettings):
    plt.figure(figsize=(8, 8))
    plt.title("Spirals")
    plt.scatter(
        data.x[:, 0],
        data.x[:, 1],
        c=data.y,
        s=40,
        cmap=plt.cm.coolwarm,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "data.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_boundry(model: MLP, data: Data, settings: PlottingSettings):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Boundary")

    feature_1, feature_2 = np.meshgrid(
        np.linspace(-15, 15),
        np.linspace(-15, 15),
    )
    grid = jnp.asarray(np.vstack([feature_1.ravel(), feature_2.ravel()]).T)
    y_pred = np.reshape(model(grid), feature_1.shape)
    display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_pred)
    display.plot()
    display.ax_.scatter(data.x[:, 0], data.x[:, 1], c=data.y, edgecolor="black")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "boundry.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
