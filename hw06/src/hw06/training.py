import optax
import structlog
from flax import nnx

from .model import Decoder
from .config import TrainingSettings
from .data import Data

log = structlog.get_logger()


def loss_fn(model: Decoder, batch: tuple) -> float:
    x, y = batch
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()
    return loss


@nnx.jit
def train_step(
    model: Decoder,
    optimizer: nnx.Optimizer,
    batch,
):
    """Performs a single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


def train(
    model: Decoder,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())

    for step in range(settings.train_steps):
        batch = data.get_batch("train")
        loss = train_step(model, optimizer, batch)

        if step % 500 == 0:
            log.info(
                "Training progress",
                step=step,
                training_loss=float(loss),
            )

        if step >= settings.train_steps:
            break

    log.info("Training finished")
