import jax.numpy as jnp
import numpy as np
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

    y_hat = jnp.clip(model(x), 1e-7, 1 - 1e-7)

    def loss_fn(model: MLP):
        # in optax they use raw logit outputs and do log_sigmoid
        y_hat = jnp.clip(model(x), 1e-7, 1 - 1e-7)
        log_p = jnp.log(y_hat)
        log_not_p = jnp.log(1 - y_hat)
        cross_entropy_loss = y * log_p + (1 - y) * log_not_p
        return -jnp.mean(cross_entropy_loss), (y_hat, x, y)

    grad_fn = nnx.grad(loss_fn, has_aux=True)
    grads, (y_hat, _, _) = grad_fn(model)
    optimizer.update(model, grads)

    return loss_fn(model)[0], grads, y_hat, x, y


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
        loss, grads, y_pred, input_x, input_y = train_step(model, optimizer, x, y)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()

        if i % 10 == 0:
            log.debug("Input x (Sample Batch)", x=x)
            log.debug("Input y (True Labels)", y=y)
            log.debug("Input y (Predicted Labels)", y_pred=y_pred)
            log.debug("Grads", grads=grads)
            log.debug(f"Loss at timestep: {i}", loss=loss)

    log.info("Training finished")
