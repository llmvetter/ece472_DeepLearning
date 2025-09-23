from dataclasses import dataclass
import tensorflow_datasets as tfds
import tensorflow as tf


@dataclass
class Data:
    def __init__(
        self,
        rng: tf.random.Generator,
        batch_size: int = 32,
        train_steps: int = 1000,
        val_split: float = 0.2,
    ) -> None:
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.split = int((1 - val_split) * 100)
        self._rng = rng

    def load(self) -> None:
        # load
        train_ds, val_ds, test_ds = tfds.load(
            "mnist",
            split=[
                f"train[:{self.split}%]",
                f"train[{self.split}%:]",
                "test",
            ],
        )

        # transform
        train_ds = train_ds.map(
            lambda sample: {
                "image": tf.cast(sample["image"], tf.float32) / 255,
                "label": sample["label"],
            }
        )
        val_ds = val_ds.map(
            lambda sample: {
                "image": tf.cast(sample["image"], tf.float32) / 255,
                "label": sample["label"],
            }
        )
        test_ds = test_ds.map(
            lambda sample: {
                "image": tf.cast(sample["image"], tf.float32) / 255,
                "label": sample["label"],
            }
        )

        # prepare (from jax docu)
        train_ds = train_ds.repeat().shuffle(1024)
        train_ds = train_ds.shuffle(buffer_size=1024, seed=42)
        self.train_ds = (
            train_ds.batch(
                batch_size=self.batch_size,
                drop_remainder=True,
            )
            .take(self.train_steps)
            .prefetch(1)
        )
        self.val_ds = val_ds.batch(
            batch_size=self.batch_size,
            drop_remainder=True,
        ).prefetch(1)

        self.test_ds = test_ds.batch(
            self.batch_size,
            drop_remainder=True,
        ).prefetch(1)
