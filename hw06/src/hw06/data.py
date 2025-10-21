from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp


class Data:
    def __init__(
        self,
        batch_size: int = 32,
        val_split: float = 0.2,
        context_length: int = 8,
        *,
        key: nnx.Rngs,
    ) -> None:
        self.key = key
        self.context_length = context_length
        self.batch_size = batch_size
        self.split = float(1 - val_split)
        self.file_path = "/home/lenni/projects/deepl/hw06/data/input.txt"

    def load(self) -> None:
        # load
        with open(self.file_path, "r") as f:
            self.data = f.read()

        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, li: list[str]) -> str:
        return "".join([self.itos[i] for i in li])

    def preprocess(self) -> None:
        n = len(self.data)
        train_data = self.data[: int(n * self.split)]
        val_data = self.data[int(n * self.split) :]
        self.train_idxs = jnp.array(self.encode(train_data), dtype=jnp.uint16)
        self.val_idxs = jnp.array(self.encode(val_data), dtype=jnp.uint16)

    def get_batch(self, split: Literal["train", "eval"]) -> jnp.ndarray:
        data = self.train_idxs if split == "train" else self.val_idxs
        ix = jax.random.randint(
            key=self.key,
            shape=(self.batch_size,),
            minval=0,
            maxval=len(data) - self.context_length,
        )
        x = jnp.stack([data[i : i + self.context_length] for i in ix])
        y = jnp.stack([data[i + 1 : i + self.context_length + 1] for i in ix])
        return x, y
