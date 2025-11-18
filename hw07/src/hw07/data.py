from typing import Literal
from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_samples: int = 500
    sigma: float = 0.2
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    logits: np.ndarray = field(init=False)

    def __post_init__(
        self,
        rng: np.random.Generator,
    ) -> None:
        self.index = np.arange(self.num_samples * 2)
        self.x = np.zeros((self.num_samples * 2, 2))
        self.y = np.zeros(self.num_samples * 2)
        self.logits = np.zeros(self.num_samples * 2)
        self._rng = rng

    def sample(self):
        num_classes = 2
        turns = 2

        # angle range for number of turns
        base_angle = np.linspace(
            0.5,
            turns * 2 * np.pi,
            self.num_samples,
        )

        for i in range(num_classes):
            noise = self._rng.normal(
                size=self.num_samples,
                scale=self.sigma,
            )

            offset_angle = base_angle + (i * np.pi)

            noisy_angle = offset_angle + noise

            # polar coordinates (r, angle) to cartesian (x, y)
            ix = range(self.num_samples * i, self.num_samples * (i + 1))
            self.x[ix] = np.c_[
                base_angle * np.sin(noisy_angle),
                base_angle * np.cos(noisy_angle),
            ]
            self.y[ix] = i

    def get_batch(
        self,
        batch_size: int,
        batch_type: Literal["logits", "targets"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""

        choices = self._rng.choice(self.index, size=batch_size)

        if batch_type == "logits":
            return self.x[choices], self.logits[choices]
        if batch_type == "targets":
            return self.x[choices], self.y[choices].flatten().reshape(-1, 1)
        else:
            return NotImplemented
