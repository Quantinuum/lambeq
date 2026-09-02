# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PytorchTrainer
==============
A trainer that wraps the training loop of a :py:class:`ClassicalModel`.

"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch

from lambeq.core.globals import VerbosityLevel
from lambeq.training.checkpoint import Checkpoint
from lambeq.training.pytorch_model import PytorchModel
from lambeq.training.trainer import EvalFuncT, Trainer
from lambeq.typing import StrPathT


class PytorchTrainer(Trainer):
    """A PyTorch trainer for the classical pipeline."""

    model: PytorchModel

    def __init__(self,
                 model: PytorchModel,
                 loss_function: Callable[..., torch.Tensor],
                 epochs: int,
                 optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
                 scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None
                 = None,
                 learning_rate: float = 1e-3,
                 device: int = -1,
                 *,
                 optimizer_args: dict[str, Any] | None = None,
                 scheduler_args: dict[str, Any] | None = None,
                 evaluate_functions: Mapping[str, EvalFuncT] | None = None,
                 evaluate_on_train: bool = True,
                 use_tensorboard: bool = False,
                 log_dir: StrPathT | None = None,
                 from_checkpoint: bool = False,
                 verbose: str = VerbosityLevel.TEXT.value,
                 seed: int | None = None) -> None:
        """Initialise a :py:class:`.Trainer` instance using the PyTorch
        backend.

        Parameters
        ----------
        model : :py:class:`.PytorchModel`
            A lambeq Model using PyTorch for tensor computation.
        loss_function : callable
            A PyTorch loss function from `torch.nn`.
        epochs : int
            Number of training epochs.
        optimizer : torch.optim.Optimizer, default: torch.optim.AdamW
            A PyTorch optimizer from `torch.optim`.
        scheduler : torch.optim.lr_scheduler, default: None
            A PyTorch scheduler for the learning rate, from
            `torch.optim.lr_scheduler`.
        learning_rate : float, default: 1e-3
            The learning rate provided to the optimizer for training.
        device : int, default: -1
            CUDA device ID used for tensor operation speed-up.
            A negative value uses the CPU.
        optimizer_args : dict of str to Any, optional
            Any extra arguments to pass to the optimizer.
        scheduler_args : dict of str to Any, optional
            Any extra arguments to pass to the scheduler.
        evaluate_functions : mapping of str to callable, optional
            Mapping of evaluation metric functions from their names.
            Structure [{"metric": func}].
            Each function takes the prediction "y_hat" and the label
            "y" as input.
            The validation step calls "func(y_hat, y)".
        evaluate_on_train : bool, default: True
            Evaluate the metrics on the train dataset.
        use_tensorboard : bool, default: False
            Use Tensorboard for visualisation of the training logs.
        log_dir : str or PathLike, optional
            Location of model checkpoints (and tensorboard log).
            Default is `runs/**CURRENT_DATETIME_HOSTNAME**`.
        from_checkpoint : bool, default: False
            Starts training from the checkpoint, saved in the log_dir.
        verbose : str, default: 'text',
            See :py:class:`VerbosityLevel` for options.
        seed : int, optional
            Random seed.

        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        super().__init__(model,
                         loss_function,
                         epochs,
                         evaluate_functions,
                         evaluate_on_train,
                         use_tensorboard,
                         log_dir,
                         from_checkpoint,
                         verbose,
                         seed)

        self.backend = 'pytorch'
        self.device = torch.device('cpu' if device < 0 else f'cuda:{device}')
        if device >= 0:
            torch.set_default_tensor_type(  # pragma: no cover
                    'torch.cuda.FloatTensor')

        optimizer_args = dict(optimizer_args or {})
        if learning_rate is not None:
            optimizer_args['lr'] = learning_rate
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)

        scheduler_args = dict(scheduler_args or {})
        self.scheduler = (scheduler(self.optimizer, **scheduler_args)
                          if scheduler is not None else None)

        self.model.to(self.device)

    def _add_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Add any additional information to the training checkpoint.

        These might include model-specific information like the random
        state of the backend or the state of the optimizer.

        Use `checkpoint.add_many()` to add multiple items.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            The checkpoint to add information to.

        """
        checkpoint.add_many(
            {'torch_random_state': torch.get_rng_state(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'scheduler_state_dict': self.scheduler.state_dict()
                if self.scheduler is not None
                else None})

    def _load_extra_checkpoint_info(self, checkpoint: Checkpoint) -> None:
        """Load additional checkpoint information.

        This includes data previously added by
        `_add_extra_checkpoint_info()`.

        Parameters
        ----------
        checkpoint : :py:class:`.Checkpoint`
            Mapping containing the checkpoint information.

        """
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        if (checkpoint['scheduler_state_dict'] is not None
                and self.scheduler is not None):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def validation_step(
            self,
            batch: tuple[list[Any], torch.Tensor]) -> tuple[torch.Tensor,
                                                            float]:
        """Perform a validation step.

        Parameters
        ----------
        batch : tuple of list and torch.Tensor
            Current batch.

        Returns
        -------
        Tuple of torch.Tensor and float
            The model predictions and the calculated loss.

        """
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)
            loss = self.loss_function(y_hat, y.to(self.device))
        return y_hat, loss.item()

    def training_step(
            self,
            batch: tuple[list[Any], torch.Tensor]) -> tuple[torch.Tensor,
                                                            float]:
        """Perform a training step.

        Parameters
        ----------
        batch : tuple of list and torch.Tensor
            Current batch.

        Returns
        -------
        Tuple of torch.Tensor and float
            The model predictions and the calculated loss.

        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y.to(self.device))
        self.train_costs.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y_hat, loss.item()

    def post_epoch_step(self, epoch: int, loss: float):
        # Step the scheduler if present
        if self.scheduler is not None:
            # Plateau scheduler wants to know about the loss
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss, epoch=epoch)
            else:
                self.scheduler.step(epoch=epoch)
