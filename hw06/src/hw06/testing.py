import jax.numpy as jnp
from flax import nnx
import jax


def test_global_causal_masking(
    model: nnx.Module,
    idx_sequence: jnp.ndarray,
    targets: jnp.ndarray,
):
    B, T = idx_sequence.shape
    model.eval()
    token_embeds = model.token_embedding_table(idx_sequence)
    pos_embeds = model.position_embed_table(jnp.arange(T))
    x = token_embeds + pos_embeds

    def loss_fn(x):
        z = model.blocks(x)
        return jnp.mean((z - targets) ** 2)

    grads = jax.grad(loss_fn)(x)
    grad_mag = jnp.abs(grads).mean(axis=-1)[0]
    return grad_mag


def test_single_token_causality(
    model: nnx.Module,
    idx_sequence: jnp.ndarray,
    targets: jnp.ndarray,
    target_token_idx: int,
):
    model.eval()
    B, T = idx_sequence.shape
    token_embeds = model.token_embedding_table(idx_sequence)
    pos_embeds = model.position_embed_table(jnp.arange(T))
    x = token_embeds + pos_embeds

    def loss_fn(x):
        z = model.blocks(x)
        z_t = z[:, target_token_idx, :]
        y_t = targets[:, target_token_idx, :]
        loss = jnp.mean((z_t - y_t) ** 2)
        return loss

    grads = jax.grad(loss_fn)(x)
    grad_mag = jnp.abs(grads).mean(axis=-1)[0]

    return grad_mag
