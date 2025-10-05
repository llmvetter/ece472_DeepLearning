from dataclasses import dataclass
import tensorflow_datasets as tfds
import tensorflow as tf


@dataclass
class Data:
    def __init__(
        self,
        rng: tf.random.Generator,
        dataset: str = "cifar10",
        batch_size: int = 256,
        train_steps: int = 5000,
        val_split: float = 0.2,
    ) -> None:
        self._rng = rng
        self.dataset = dataset
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.split = int((1 - val_split) * 100)

    def _preprocess(self, sample):
        image = sample["image"]
        image = tf.cast(image, tf.float32) / 255.0

        # augment
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, max_delta=0.08)

        return {
            "image": image,
            "label": sample["label"],
        }

    def load(self) -> None:
        # load
        train_ds, val_ds, test_ds = tfds.load(
            self.dataset,
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
        self.train_ds = (
            train_ds.repeat()
            .shuffle(buffer_size=10000, reshuffle_each_iteration=True, seed=42)
            .batch(
                batch_size=self.batch_size,
                drop_remainder=True,
            )
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
