import structlog
import jax
from flax import nnx


from .logging import configure_logging
from .config import load_settings
from .model import Decoder, count_params


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)

    decoder = Decoder(
        vocab_size=settings.training.vocab_size,
        n_embed=settings.training.n_embed,
        n_blocks=settings.training.n_blocks,
        block_size=settings.training.block_size,
        n_heads=settings.training.n_heads,
        rngs=nnx.Rngs(params=key),
    )
    log.debug("Initial model", model=decoder)
    log.info("Total trainable parameters", n_params=count_params(decoder))
