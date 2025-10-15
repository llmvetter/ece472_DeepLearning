import optax
import structlog
from flax import nnx

from .model import MLP

log = structlog.get_logger()


def loss_fn(model: MLP, batch) -> float:
    logits = model(batch["content"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    batch,
):
    """Performs a single training step."""

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MLP, metrics: nnx.MultiMetric, batch):
    logits = model(batch["content"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


def test(model: MLP, metrics: nnx.MultiMetric, ds):
    model.eval()
    for eval_batch in ds.as_numpy_iterator():
        eval_step(model, metrics, eval_batch)
    accuracy = metrics.compute().get("accuracy")
    log.info("Final Test Accuracy", metric=accuracy)


def train(
    model: MLP,
    optimizer: nnx.Optimizer,
    train_ds,
    eval_ds,
    metrics: nnx.MultiMetric,
) -> float:
    """Train the model using SGD."""

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        model.train()
        loss = train_step(model, optimizer, batch)

        if step % 100 == 0 and step > 0:
            log.info(
                "Training progress",
                step=step,
                training_loss=float(loss),
            )
    # eval model per fold
    model.eval()
    for eval_batch in eval_ds.as_numpy_iterator():
        eval_step(model, metrics, eval_batch)
    metric_values = metrics.compute()
    for metric, value in metric_values.items():
        log.info(f"eval_{metric}", metric=value)

    # collect fold accuracy
    accuracy = metric_values.get("accuracy")
    metrics.reset()
    log.info("Fold Finished.")
    return accuracy
