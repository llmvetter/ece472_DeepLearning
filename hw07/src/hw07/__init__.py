import jax
import numpy as np
import structlog
import optax
from flax import nnx

from .data import Data
from .model import MLP, AutoEncoder
from .config import load_settings
from .training import train
from .logging import configure_logging
from .plotting import plot_boundry, plot_feature_activation
from .analytics import get_freqs


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, mlp_key, ae_key, plot_key = jax.random.split(key, 4)
    np_rng = np.random.default_rng(np.array(data_key))

    log.debug("Generating Data")

    data = Data(
        rng=np_rng,
        num_samples=settings.data.num_samples,
    )
    data.sample()

    model = MLP(
        num_inputs=settings.mlp_training.num_inputs,
        layer_width=settings.mlp_training.layer_width,
        layer_depth=settings.mlp_training.layer_depth,
        num_outputs=settings.mlp_training.num_outputs,
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
        rngs=nnx.Rngs(params=mlp_key),
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(settings.mlp_training.learning_rate),
        wrt=nnx.Param,
    )
    # train model wrt to targets
    train(model, optimizer, data, settings.mlp_training, wrt="targets")

    # plot decision boundry
    plot_boundry(data=data, model=model, settings=settings.plotting)

    # generate logits data
    ae_data = Data(
        rng=np_rng,
        sigma=0.3,
        num_samples=settings.data.num_samples * settings.ae_training.sample_multi,
    )
    ae_data.logits = model(ae_data.x, get_logits=True)

    # init autoencoder
    auto_encoder = AutoEncoder(
        d_mlp=settings.mlp_training.layer_width,
        d_enc=settings.ae_training.layer_width,
        rngs=nnx.Rngs(params=ae_key),
    )

    optimizer = nnx.Optimizer(
        auto_encoder,
        optax.adam(settings.ae_training.learning_rate),
        wrt=nnx.Param,
    )

    # train autoencoder wrt to logits
    train(auto_encoder, optimizer, ae_data, settings.ae_training, wrt="logits")

    # insert auto encoder into MLP
    model.ae = auto_encoder
    plot_boundry(
        data=data, model=model, settings=settings.plotting, name="Reconstructed_"
    )

    # Analyze the result
    get_freqs(model, ae_data)
    plot_feature_activation(model, data, plot_key, settings.plotting, n_plots=30)
