import jax.numpy as jnp
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


def plot_boundry(model: MLP, data: Data, settings: PlottingSettings):
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
    output_path = settings.output_dir / "boundry.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
