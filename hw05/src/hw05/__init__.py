import jax
import structlog
import optax
from flax import nnx

from .config import load_settings
from .data import Data
from .model import MLP
from .training import train
from .logging import configure_logging


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)

    log.debug("Generating Data")

    data = Data()
    data.load()

    model = MLP(
        num_inputs=settings.training.vector_dim,
        layer_width=settings.training.layer_width,
        layer_depth=settings.training.layer_depth,
        num_outputs=settings.training.num_outputs,
        output_activation=nnx.identity,
        rngs=nnx.Rngs(params=model_key),
    )

    log.debug("Initial model", model=model)

    optimizer = nnx.Optimizer(
        model,
        optax.adam(
            settings.training.learning_rate,
            settings.training.momentum,
        ),
        wrt=nnx.Param,
    )

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    train(model, optimizer, data, settings.training, metrics)
