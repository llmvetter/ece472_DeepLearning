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

    def _preprocess(self, sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        return {
            "image": image,
            "label": sample["label"],
        }

    def load(self) -> None:
        # load
        train_ds, val_ds, test_ds = tfds.load(
            "cifar10",
            split=[
                f"train[:{self.split}%]",
                f"train[{self.split}%:]",
                "test",
            ],
        )

        # transform
        train_ds = train_ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # prepare (from jax docu)
        train_ds = train_ds.shuffle(buffer_size=1024, seed=42)
        self.train_ds = (
            train_ds.batch(
                batch_size=self.batch_size,
                drop_remainder=True,
            )
            .take(self.train_steps)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val_ds = val_ds.batch(
            batch_size=self.batch_size,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)

        self.test_ds = test_ds.batch(
            self.batch_size,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)
