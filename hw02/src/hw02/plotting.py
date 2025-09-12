import matplotlib
import matplotlib.pyplot as plt
import structlog

from .config import PlottingSettings
from .data import Data

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
