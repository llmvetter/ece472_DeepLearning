import jax
import numpy as np
import structlog
import optax
from flax import nnx

from .config import load_settings
from .data import Data
from .model import MLP, AutoEncoder
from .training import train
from .logging import configure_logging
from .plotting import plot_boundry


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, mlp_key, ae_key = jax.random.split(key, 3)
    np_rng = np.random.default_rng(np.array(data_key))

    log.debug("Generating Data")

    data = Data(
        rng=np_rng,
        num_samples=settings.data.num_samples,
    )
    data.sample()

    model = MLP(
        num_inputs=settings.training.num_inputs,
        layer_width=settings.training.layer_width,
        layer_depth=settings.training.layer_depth,
        num_outputs=settings.training.num_outputs,
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
        rngs=nnx.Rngs(params=mlp_key),
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(settings.training.learning_rate),
        wrt=nnx.Param,
    )
    # train model wrt to targets
    train(model, optimizer, data, settings.training, wrt="targets")

    # plot decision boundry
    plot_boundry(data=data, model=model, settings=settings.plotting)

    # generate logits data
    data.logits = model(data.x, get_logits=True)

    # init autoencoder
    auto_encoder = AutoEncoder(
        d_mlp=settings.training.layer_width,
        d_enc=1000,
        rngs=nnx.Rngs(params=ae_key),
    )

    optimizer = nnx.Optimizer(
        auto_encoder,
        optax.adam(settings.training.learning_rate),
        wrt=nnx.Param,
    )

    # train autoencoder wrt to logits
    train(auto_encoder, optimizer, data, settings.training, wrt="logits")
