import jax
import jax.numpy as jnp
from typing import Literal
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import MLPSettings, AESettings
from .data import Data
from .model import MLP, AutoEncoder

log = structlog.get_logger()


def mlp_loss(model: MLP, x: jax.Array, y: jax.Array) -> float:
    # in optax they use raw logit outputs and do log_sigmoid
    logits = model(x)
    bce = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            logits=logits,
            labels=y,
        )
    )
    return bce


def ae_loss(model: AutoEncoder, x: jax.Array, y: jax.Array) -> float:
    # y are logits, x are coordinates
    x_rcst = model(y)
    l2_loss = jnp.sum(jnp.power((x_rcst - y), 2))
    activations = model.act(x_rcst @ model.W_enc + model.b_enc)
    l1_loss = model.l1_coeff * jnp.mean(
        jnp.sum(
            jnp.abs(activations),
            axis=1,
        ),
    )
    loss = l2_loss + l1_loss

    return loss


@nnx.jit
def train_step(
    model: MLP | AutoEncoder,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    """Performs a single training step."""
    if isinstance(model, MLP):
        loss_fn = mlp_loss
    if isinstance(model, AutoEncoder):
        loss_fn = ae_loss
    loss, grads = nnx.value_and_grad(loss_fn, argnums=0)(model, x, y)
    optimizer.update(model, grads)

    return loss


def train(
    model: MLP | AutoEncoder,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: AESettings | MLPSettings,
    *,
    wrt: Literal["targets", "logits"],
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(
            batch_size=settings.batch_size,
            batch_type=wrt,
        )

        x, y = jnp.asarray(x_np), jnp.asarray(y_np)
        loss = train_step(model, optimizer, x, y)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()

    log.info("Training finished")
