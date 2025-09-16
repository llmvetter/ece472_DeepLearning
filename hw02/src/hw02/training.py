import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import MLP

log = structlog.get_logger()


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    """Performs a single training step."""

    def loss_fn(model: MLP):
        # in optax they use raw logit outputs and do log_sigmoid
        logits = model(x)
        bce = jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                logits=logits,
                labels=y,
            )
        )
        return bce

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


def train(
    model: MLP,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)

        # array conversion
        x, y = jnp.asarray(x_np), jnp.asarray(y_np).reshape(-1, 1)
        loss = train_step(model, optimizer, x, y)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()

        if i % 10 == 0:
            log.debug(f"Loss at timestep: {i}", loss=loss)

    log.info("Training finished")
