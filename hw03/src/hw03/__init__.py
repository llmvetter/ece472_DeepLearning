import jax
import tensorflow as tf
import structlog
import optax
from flax import nnx

from .config import load_settings
from .data import Data
from .model import Classifier
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
    data_key, model_key = jax.random.split(key)
    tf_rng = tf.random.Generator.from_seed(data_key)

    log.debug("Generating Data")
    data = Data(
        rng=tf_rng,
        val_split=settings.data.val_split,
        batch_size=settings.training.batch_size,
        train_steps=settings.training.train_steps,
    )
    data.load()

    model = Classifier(
        input_depth=settings.training.input_depth,
        layer_depths=settings.training.layer_depths,
        num_classes=settings.training.num_classes,
        layer_kernel_sizes=settings.training.layer_kernel_sizes,
        rngs=nnx.Rngs(model_key),
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

    # Train and eval on train and test
    train(model, optimizer, data, settings.training, metrics)

    # Eval model on testset, once
    test(model, data)
