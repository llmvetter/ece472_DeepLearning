import jax
import structlog
import optax
import numpy as np
from flax import nnx

from .config import load_settings
from .data import Data
from .model import MLP
from .model import Embedder
from .training import train, test
from .logging import configure_logging


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)

    # Load and preprocess data
    log.debug("Generating Data")
    embedder = Embedder()
    data = Data(
        embedder,
        n_splits=settings.data.folds,
    )
    data.load()

    # start k-fold training
    log.info("Starting training", **settings.model_dump())
    accs = []
    for idx, train_ds, eval_ds in data.kfolds():
        log.info(f"Starting training on fold: {idx}")

        model = MLP(
            num_inputs=settings.training.vector_dim,
            layer_width=settings.training.layer_width,
            layer_depth=settings.training.layer_depth,
            num_outputs=settings.training.num_outputs,
            output_activation=nnx.identity,
            rngs=nnx.Rngs(params=key),
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
        acc = train(
            model=model,
            optimizer=optimizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            metrics=metrics,
        )
        accs.append(acc)

    avg_accuracy = np.mean(accs)
    std_accuracy = np.std(accs)
    log.info(
        f"Average Accuracy across {data.splits} folds: {avg_accuracy:.4f} ± {std_accuracy:.4f}"
    )

    test(model=model, metrics=metrics, ds=data.test_ds)
