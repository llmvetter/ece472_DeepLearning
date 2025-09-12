import jax
import numpy as np
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .model import MLP
from .logging import configure_logging
from .plotting import plot_data


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    log.debug("Generating Data")

    data = Data(
        rng=np_rng,
        num_samples=settings.data.num_samples,
    )
    data.sample()
    log.debug("Datapoints X", data=len(data.x))
    log.debug("Datapoints Y", data=len(data.y))

    plot_data(data, settings.plotting)

    model = MLP(
        num_inputs=settings.training.num_inputs,
        layer_width=settings.training.layer_width,
        layer_depth=settings.training.layer_depth,
        num_outputs=settings.training.num_outputs,
        hidden_activation=nnx.relu,
        output_activation=nnx.sigmoid,
        rngs=nnx.Rngs(params=model_key),
    )

    log.debug("Initial model", model=model)
    log.debug("Model Shape", model=nnx.display(model))
