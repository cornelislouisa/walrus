import logging
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class StagedLRScheduler(_LRScheduler):
    """
    Learning rate scheduler for staged fine-tuning.
    - New parameters: Follow the main LR scheduler from the start
    - Common parameters: LR=0 for warmup_epochs, then follow same schedule as new parameters
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        main_scheduler: _LRScheduler,
        warmup_epochs: int = 5,
        last_epoch: int = -1,
    ):
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.param_group_info = {}

        # Identify parameter groups by position and initial LR
        # Group 0: new parameters (LR > 0)
        # Group 1+: common parameters (LR = 0) if they exist
        for i, group in enumerate(optimizer.param_groups):
            initial_lr = group["lr"]
            is_common_group = initial_lr == 0.0  # Common params start with LR=0

            self.param_group_info[i] = {
                "type": "common" if is_common_group else "new",
                "group_index": i,
            }

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # Get LRs from the main scheduler
        main_lrs = self.main_scheduler.get_lr()

        lrs = []
        new_param_lr = None

        # First pass: find the LR for new parameters
        for i, group in enumerate(self.optimizer.param_groups):
            group_info = self.param_group_info[i]
            if group_info["type"] == "new":
                new_param_lr = main_lrs[i] if i < len(main_lrs) else main_lrs[0]
                break

        # If no new parameter LR found, use the first one from main scheduler
        if new_param_lr is None:
            new_param_lr = main_lrs[0] if main_lrs else 0.0

        # Second pass: assign LRs
        for i, group in enumerate(self.optimizer.param_groups):
            group_info = self.param_group_info[i]

            if group_info["type"] == "new":
                # New parameters use their LR from main scheduler
                lr = main_lrs[i] if i < len(main_lrs) else main_lrs[0]
                lrs.append(lr)
            else:  # common parameters
                if self.last_epoch <= self.warmup_epochs:
                    # Keep at LR=0 during warmup
                    lrs.append(0.0)
                else:
                    # After warmup, use same LR as new parameters
                    lrs.append(new_param_lr)

        return lrs

    def step(self, epoch: Optional[int] = None):
        # Store old LRs for logging
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Step the main scheduler first
        self.main_scheduler.step(epoch)

        # Then step this scheduler
        super().step(epoch)

        # Log when common parameters start training
        for i, group in enumerate(self.optimizer.param_groups):
            group_info = self.param_group_info[i]
            if group_info["type"] == "common":
                old_lr = old_lrs[i]
                new_lr = group["lr"]

                if old_lr == 0.0 and new_lr > 0.0:
                    logger.info(
                        f"Epoch {self.last_epoch + 1}: Common parameters starting to train with LR={new_lr:.6f}"
                    )

    def state_dict(self):
        """Return the state of the scheduler."""
        state = super().state_dict()
        state["main_scheduler"] = self.main_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load the state of the scheduler."""
        main_scheduler_state = state_dict.pop("main_scheduler", None)
        super().load_state_dict(state_dict)
        if main_scheduler_state is not None:
            self.main_scheduler.load_state_dict(main_scheduler_state)
