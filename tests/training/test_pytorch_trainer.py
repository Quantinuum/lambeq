from math import ceil
import time

import torch
import numpy as np
import pytest

from lambeq.backend.grammar import Cup, Id, Word
from lambeq.backend.tensor import Dim

from lambeq import (AtomicType, Dataset, PytorchModel, PytorchTrainer,
                    SpiderAnsatz)

N = AtomicType.NOUN
S = AtomicType.SENTENCE
EPOCHS = 1

sig = torch.sigmoid
acc = lambda y_hat, y: torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2


train_diagrams = [
    (Word("Alice", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Alice", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("runs", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
]
train_targets = [[1, 0], [0, 1], [0, 1], [1, 0]]

dev_diagrams = [
    (Word("Alice", N) @ Word("eats", N >> S) >> Cup(N, N.r) @ Id(S)),
    (Word("Bob", N) @ Word("waits", N >> S) >> Cup(N, N.r) @ Id(S)),
]
dev_targets = [[0, 1], [1, 0]]
ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
train_circuits = [ansatz(d) for d in train_diagrams]
dev_circuits = [ansatz(d) for d in dev_diagrams]


@pytest.mark.parametrize("scheduler", [None, torch.optim.lr_scheduler.StepLR])
def test_trainer(tmp_path, scheduler):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)

    log_dir = tmp_path / 'test_runs'
    log_dir.mkdir()
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        scheduler=scheduler,
        scheduler_args={"step_size": 1, "gamma": 0.1},
        learning_rate=3e-3,
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    assert len(trainer.train_costs) == EPOCHS
    assert len(trainer.val_eval_results["acc"]) == EPOCHS
    assert len(trainer.train_epoch_durations) == EPOCHS
    assert len(trainer.train_durations) == EPOCHS * (
        ceil(len(train_diagrams) / train_dataset.batch_size))
    assert len(trainer.val_durations) == EPOCHS
    if scheduler is not None:
        # Expect lr to have decayed exactly once, up to float error = 0.1 * 3e-3
        assert torch.allclose(
            torch.tensor(trainer.scheduler.get_last_lr()),
            torch.tensor(3e-4)
        )


def test_restart_training(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
    log_dir.mkdir()
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset)

    model_new = PytorchModel()
    trainer_restarted = PytorchTrainer(
        model=model_new,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS+1,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        from_checkpoint=True,
        verbose='suppress',
        seed=0
    )

    trainer_restarted.fit(train_dataset, val_dataset)

    model_uninterrupted = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    trainer_uninterrupted = PytorchTrainer(
        model=model_uninterrupted,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=EPOCHS+1,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        verbose='suppress',
        seed=0
    )

    trainer_uninterrupted.fit(train_dataset, val_dataset)

    assert len(trainer_restarted.train_costs) == EPOCHS+1
    assert len(trainer_restarted.val_costs) == EPOCHS+1
    assert len(trainer_restarted.val_eval_results["acc"]) == EPOCHS+1
    assert len(trainer_restarted.train_eval_results["acc"]) == EPOCHS+1
    for a, b in zip(trainer_restarted.train_costs, trainer_uninterrupted.train_costs):
        assert np.isclose(a, b)
    for a, b in zip(model_new.weights, model_uninterrupted.weights):
        assert torch.allclose(a, b)


def test_evaluation_skipping(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
    epochs = 4
    eval_step = 2
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=epochs,
        evaluate_functions={"acc": acc},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)

    trainer.fit(train_dataset, val_dataset, eval_interval=eval_step)

    assert len(trainer.train_costs) == epochs
    assert len(trainer.train_durations) == epochs
    assert len(trainer.val_eval_results["acc"]) == ceil(epochs/eval_step)
    assert len(trainer.val_durations) == ceil(epochs/eval_step)
    assert len(trainer.train_epoch_durations) == epochs


def test_early_stopping(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
    epochs = 100
    inc_function = lambda _, __: time.time() * 1e-10
    eval_step = 1
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=epochs,
        evaluate_functions={"acc": inc_function},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)


    trainer.fit(train_dataset, val_dataset,
                eval_interval=eval_step,
                early_stopping_criterion="acc",
                early_stopping_interval=5)


    assert len(trainer.val_eval_results["acc"]) == 6
    assert len(trainer.train_costs) == 6
    assert len(trainer.train_durations) == 6
    assert len(trainer.val_durations) == 6
    assert len(trainer.train_epoch_durations) == 6


def test_early_stopping_max(tmp_path):
    model = PytorchModel.from_diagrams(train_circuits + dev_circuits)
    log_dir = tmp_path / 'test_run'
    epochs = 100
    dec_function = lambda _, __: -time.time() * 1e-10
    eval_step = 1
    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=3e-3,
        epochs=epochs,
        evaluate_functions={"acc": dec_function},
        evaluate_on_train=True,
        use_tensorboard=True,
        log_dir=log_dir,
        verbose='suppress',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_targets)
    val_dataset = Dataset(dev_circuits, dev_targets)


    trainer.fit(train_dataset, val_dataset,
                eval_interval=eval_step,
                early_stopping_criterion="acc",
                early_stopping_interval=10,
                minimize_criterion=False)


    assert len(trainer.val_eval_results["acc"]) == 11
    assert len(trainer.train_costs) == 11
    assert len(trainer.train_durations) == 11
    assert len(trainer.val_durations) == 11
    assert len(trainer.train_epoch_durations) == 11
