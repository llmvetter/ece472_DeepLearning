import structlog
import jax
from flax import nnx
import jax.numpy as jnp
import optax


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
from .training import train
from .testing import (
    test_global_causal_masking,
    test_single_token_causality,
)


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG & Globals
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key, dropout_key = jax.random.split(key, num=3)
    head_size = settings.training.n_embed // settings.training.n_heads

    # Dummy input: B,T,C = 16, 32, 64
    x_dummy = jnp.ones(
        (
            settings.training.batch_size,
            settings.training.context_length,
            settings.training.n_embed,
        )
    )
    B, T, C = x_dummy.shape
    log.info("===== Initializing Test Suite =====")
    ## Test Feed Forward
    ffw = FeedForward(
        n_emb=settings.training.n_embed,
        rngs=nnx.Rngs(params=model_key, dropout=dropout_key),
    )
    output = ffw(x_dummy)
    test_out = x_dummy.shape == output.shape
    log.info("Test: Feed Forward Output shape matches input shape", test=test_out)

    ## Test Attention Head
    att_head = Head(
        head_size=head_size,
        n_embed=settings.training.n_embed,
        rngs=nnx.Rngs(params=model_key, dropout=dropout_key),
    )
    output = att_head(x_dummy)
    test_out = output.shape == (B, T, head_size)
    log.info(
        "Test: Attention Output shape matches expectation B, T, head_size",
        test=test_out,
    )

    ## Test Multi Head Attention
    mh_att = MultiHeadAttention(
        n_heads=settings.training.n_heads,
        head_size=head_size,
        n_embed=settings.training.n_embed,
        rngs=nnx.Rngs(params=model_key, dropout=dropout_key),
    )
    output = mh_att(x_dummy)
    test_out = x_dummy.shape == output.shape
    log.info(
        "Test: Multi Headed Attention Output shape matches input shape", test=test_out
    )

    ## Test Block
    block = Block(
        n_embed=settings.training.n_embed,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=model_key, dropout=dropout_key),
    )
    output = block(x_dummy)
    test_out = x_dummy.shape == output.shape
    log.info("Test: Attention Block Output shape matches input shape", test=test_out)

    ## Test Decoder Causal Mapping
    idx_sequence = jax.random.randint(
        model_key,
        shape=(
            settings.training.batch_size,
            settings.training.context_length,
        ),
        minval=1,
        maxval=settings.training.vocab_size,
        dtype=jnp.int32,
    )
    targets = jnp.ones(
        (
            settings.training.batch_size,
            settings.training.context_length,
            settings.training.n_embed,
        )
    )
    decoder = Decoder(
        vocab_size=settings.training.vocab_size,
        n_embed=settings.training.n_embed,
        n_blocks=settings.training.n_blocks,
        context_length=settings.training.context_length,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=model_key, dropout=dropout_key),
    )
    global_gradients = test_global_causal_masking(
        model=decoder,
        idx_sequence=idx_sequence,
        targets=targets,
    )
    token_gradients = test_single_token_causality(
        model=decoder,
        idx_sequence=idx_sequence,
        target_token_idx=32,
        targets=targets,
    )
    log.info(
        "Test: Gradient wrt. to all target tokens implicates causality.",
        grad=global_gradients,
    )
    log.info(
        "Test: Gradient wrt. single target token shows masking.", grad=token_gradients
    )

    log.info("===== Initializing Training Pipeline =====")
    ## Data Loading
    data = Data(
        key=data_key,
        batch_size=settings.training.batch_size,
        val_split=settings.training.split,
    )
    data.load()
    data.preprocess()
    log.info("Vocab Loaded", chars=len(data.chars))
    log.info("Dataset Loaded", data=len(data.data))
    log.info("Train dataset", test=len(data.train_idxs))
    log.info("Eval dataset", val=len(data.val_idxs))

    optimizer = nnx.Optimizer(
        decoder,
        optax.adam(
            settings.training.learning_rate,
            settings.training.momentum,
        ),
        wrt=nnx.Param,
    )
    # Generate some random output
    decoder.eval()
    context = jnp.zeros(shape=(1, 1), dtype=jnp.int32)
    out = data.decode(decoder.generate(context, 20, temp=1.5)[0].tolist())
    log.info("Random model generated text", text=out)

    # Train
    log.info("Total trainable parameters", n_params=count_params(decoder))
    decoder.train()
    train(decoder, optimizer, data, settings.training)

    # Generate better output
    decoder.eval()
    out = data.decode(
        decoder.generate(
            idx=context,
            max_new_tokens=20,
            temp=1.5,
        )[0].tolist()
    )
    log.info("Trained model generated text", text=out)
