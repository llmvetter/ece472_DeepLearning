import jax
import tensorflow as tf
import structlog
import optax
from flax import nnx

from .config import load_settings
from .data import Data
from .model import Classifier
from .training import train
from .logging import configure_logging
from .checkpointing import Checkpointer


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

    # data
    log.debug("Generating Data")
    data = Data(
        rng=tf_rng,
        val_split=settings.data.val_split,
        batch_size=settings.training.batch_size,
        train_steps=settings.training.train_steps,
    )
    data.load()

    # model
    model = Classifier(
        input_depth=settings.training.input_depth,
        num_blocks=(3, 3, 3),
        layer_depths=settings.training.layer_depths,
        num_classes=settings.training.num_classes,
        layer_kernel_sizes=settings.training.layer_kernel_sizes,
        rngs=nnx.Rngs(model_key),
    )

    log.debug("Initial model", model=model)

    # optimizer with exonential decay schedule
    optimizer = nnx.Optimizer(
        model,
        optax.adam(
            optax.exponential_decay(
                init_value=settings.training.learning_rate,
                transition_steps=100,
                decay_rate=0.95,
            ),
            settings.training.momentum,
        ),
        wrt=nnx.Param,
    )

    # metrics
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    # checkpointer
    if settings.training.fresh_start is not False:
        checkpointer = Checkpointer(kill=False)

        # attempting to load model
        log.info("Loading latest Checkpoint...")
        abstract_model = nnx.eval_shape(
            lambda: Classifier(
                input_depth=settings.training.input_depth,
                num_blocks=(3, 3, 3),
                layer_depths=settings.training.layer_depths,
                num_classes=settings.training.num_classes,
                layer_kernel_sizes=settings.training.layer_kernel_sizes,
                rngs=nnx.Rngs(0),
            )
        )
        model = checkpointer.load(
            abstract_model=abstract_model, model_state=nnx.state(model)
        )

    else:
        checkpointer = Checkpointer(kill=True)
        log.info("Initializing fresh training start.")

    # train and eval on train and test
    train(
        model=model,
        optimizer=optimizer,
        data=data,
        settings=settings.training,
        metrics=metrics,
        checkpointer=checkpointer,
    )
    # Eval model on testset, once
    # test(model, data)
