import structlog
import jax
from flax import nnx
import jax.numpy as jnp


from .logging import configure_logging
from .config import load_settings
from .model import (
    Decoder,
    FeedForward,
    Head,
    MultiHeadAttention,
    Block,
    count_params,
)
from .data import Data


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)

    decoder = Decoder(
        vocab_size=settings.training.vocab_size,
        n_embed=settings.training.n_embed,
        n_blocks=settings.training.n_blocks,
        context_length=settings.training.context_length,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=model_key),
    )
    log.info("Total trainable parameters", n_params=count_params(decoder))

    # Dummy input: B,T,C = 16, 8, 32
    x_dummy = jnp.ones(
        (
            settings.training.batch_size,
            settings.training.context_length,
            settings.training.n_embed,
        )
    )
    B, T, C = x_dummy.shape

    ## Test Feed Forward
    ffw = FeedForward(n_emb=settings.training.n_embed, rngs=nnx.Rngs(params=key))
    output = ffw(x_dummy)
    test_out = x_dummy.shape == output.shape

    log.info("Feed Forward Output shape matches input shape", test=test_out)

    ## Test Attention Head
    att_head = Head(
        head_size=settings.training.context_length,
        n_embed=settings.training.n_embed,
        rngs=nnx.Rngs(params=model_key),
    )
    output = att_head(x_dummy)
    test_out = output.shape == (B, T, T)

    log.info("Attention Matrix retruns Shape", shape=output.shape)
    log.info("Attention Output shape matches expectation B, T, T", test=test_out)

    ## Test Multi Head Attention
    mh_att = MultiHeadAttention(
        n_heads=settings.training.n_heads,
        head_size=settings.training.context_length,
        n_embed=settings.training.n_embed,
        rngs=nnx.Rngs(params=model_key),
    )
    output = mh_att(x_dummy)
    test_out = x_dummy.shape == output.shape
    log.info("Mulit Headed Attention retruns Shape", shape=output.shape)
    log.info("Multi Headed Attention Output shape matches input shape", test=test_out)

    ## Test Block
    block = Block(
        n_embed=settings.training.n_embed,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=model_key),
    )
    output = block(x_dummy)
    test_out = x_dummy.shape == output.shape
    log.info("Attention Block Output shape matches input shape", test=test_out)

    ## Test Decoder
    idx_sequence = jax.random.randint(
        model_key, shape=(16, 8), minval=1, maxval=101, dtype=jnp.int32
    )
    decoder = Decoder(
        vocab_size=settings.training.vocab_size,
        n_embed=settings.training.n_embed,
        n_blocks=settings.training.n_blocks,
        context_length=settings.training.context_length,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=model_key),
    )
    output = decoder(idx_sequence)
    log.info("Decoder Output shape", shape=output.shape)

    data = Data(
        key=data_key,
        batch_size=settings.training.batch_size,
        val_split=settings.training.split,
    )
    data.load()
    data.preprocess()
    log.info("Vocab Loaded", chars=data.chars)
    log.info("Dataset Loaded", data=len(data.data))
    log.info("Train dataset", test=len(data.train_idxs))
    log.info("Eval dataset", val=len(data.val_idxs))

    x, y = data.get_batch("train")
    log.info("Sample batch", x_y=(x, y))
