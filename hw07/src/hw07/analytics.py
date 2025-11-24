import jax
import jax.numpy as jnp
import structlog

from .model import MLP
from .data import Data

# AutoEncoder activation frequencies
log = structlog.get_logger()


def get_freqs(
    model: MLP,
    data: Data,
    num_batches: int = 32,
) -> jax.Array:
    act_freq_scores = jnp.zeros(model.ae.d_enc)
    total = 0
    for i in range(num_batches):
        x_np, y_np = data.get_batch(
            batch_size=16,
            batch_type="logits",
        )
        _, y = jnp.asarray(x_np), jnp.asarray(y_np)
        activations = model.ae(y, get_acts=True)
        act_freq_scores += (activations > 0).sum(0)
        total += activations.shape[0]
    act_freq_scores /= total
    num_dead = jnp.mean(act_freq_scores == 0)
    log.info("Percentage of dead activations", n=f"{num_dead * 100}%")
    return act_freq_scores
