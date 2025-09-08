from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""

    rng: InitVar[np.random.Generator]
    num_samples: int = 50
    sigma: float = 0.1
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Generate synthetic data."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(
            low=0,
            high=2,
            size=(self.num_samples, 1),
        )
        self.clean_y = np.sin(2 * np.pi * self.x)
        noise = rng.normal(
            size=self.clean_y.shape,
            scale=self.sigma,
        )
        self.y = self.clean_y + noise

    def get_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
