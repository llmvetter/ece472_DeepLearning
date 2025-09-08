from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class LinearModel:
    """Represents a rbf model."""

    weights: np.ndarray
    bias: float
    mus: np.ndarray
    sigmas: np.ndarray


class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a nonlinear rbf model."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        n_kernels: int = 5,
    ) -> None:
        key = rngs.params()
        self.m = n_kernels
        self.w = nnx.Param(jax.random.normal(key, (self.m,)))
        self.b = nnx.Param(jnp.zeros((1,)))
        self.mu = nnx.Param(jax.random.normal(key, (self.m, 1)))
        self.sigma = nnx.Param(
            jax.random.uniform(
                key,
                (self.m, 1),
                minval=0.01,
                maxval=0.99,
            )
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        """for each scalar xi pass it to the m basis functions:
            compute phi for xi for each basis function shape is vector of len m
            weight each phi value with weight and sum up the weighted phis
            add bias"""

        x_expanded = jnp.expand_dims(x, axis=1)
        exponential = -jnp.square(x_expanded - self.mu.value) / jnp.square(
            self.sigma.value
        )
        phi = jnp.exp(jnp.sum(exponential, axis=-1))
        phi_sum = jnp.sum(phi * self.w.value, axis=1) + self.b.value
        return phi_sum

    @property
    def model(self) -> LinearModel:
        """Returns the underlying rbf model."""
        return LinearModel(
            weights=np.array(self.w.value).reshape([self.m]),
            bias=np.array(self.b.value).squeeze(),
            mus=np.array(self.mu.value).reshape([self.m]),
            sigmas=np.array(self.sigma.value).reshape([self.m]),
        )
