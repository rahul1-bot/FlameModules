# MIT License
#     
# Copyright (c) 2023 Rahul Sawhney
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# __author_info__: dict[str, Union[str, list[str]]] = {
#     'Name': 'Rahul Sawhney',
#     'Mail': [
#         'sawhney.rahulofficial@outlook.com', 
#         'rahulsawhney321@gmail.com'
#     ]
# }
from __future__ import annotations
from FlameModules.callbacks.callback import Callback
import torch, logging
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn

log = logging.getLogger(__name__)

# -------------------
# EarlyStopping in AI
# -------------------
#
# What is EarlyStopping?
#
#     EarlyStopping is a regularization technique used in the training of machine learning and deep learning models. 
#     The method involves monitoring a specific performance metric, usually validation loss, over epochs. Training is 
#     halted once the performance stops improving, indicating a potential onset of overfitting.

# Why do we need EarlyStopping?
#
#     1. Prevent Overfitting: Overfitting occurs when a model captures noise or random fluctuations in the 
#                             training data. This leads to excellent performance on training data but poor 
#                             generalization to unseen data. EarlyStopping prevents the model from learning 
#                             these nuances, ensuring a better balance between bias and variance.
#
#     2. Save Time and Resources: Training deep learning models, especially with large datasets, can be 
#                                 time-intensive and consume significant computational resources. EarlyStopping 
#                                 allows us to stop training once the model's performance plateaus, optimizing 
#                                 resource usage.
# 
#     3. Optimal Model Selection: During training, model performance on validation data can fluctuate. 
#                                 EarlyStopping, often combined with Model Checkpointing, ensures that the best 
#                                 model, in terms of validation performance, is saved and not necessarily the 
#                                 last one.
#
#     4. Reduces the Need for Guesswork: Setting the exact number of training epochs beforehand can be 
#                                        challenging. Too few epochs can result in an underfit model, while too 
#                                        many can lead to overfitting. EarlyStopping provides a dynamic approach 
#                                        to determine the optimal number of epochs.
#
#     5. Confidence in Model Convergence: In some cases, models may never converge or take a very long time 
#                                         to do so. With EarlyStopping, we can have more confidence in the 
#                                         model's convergence, as it stops when improvements become negligible.

class EarlyStopping(Callback):
    #@: Monitor a metric and stop training when its stops imporving. 
    # Args:
    #       * Monitor (str): Quality to be monitored
    #
    #       * min_delta (float): Minimum change in the monitored quality as an improvement, 
    #                            An absolute change of less than or equal to `min_delta`, will 
    #                            count as no improvement.
    #
    #
    #       * patience (int): Number of checks with `No Improvement` after which training will stop.
    #
    #       * verbose (bool): Verbosity mode.
    #
    #       * mode (str): One of {`min`, `max`}
    #                     - `min`: Training will stop when the quantity monitored has stopped decreasing. 
    #                     - `max`: Training will stop when the quantity monitored has stopped increasing.
    # 
    #       * strict (bool): Whether to crash the training is `monitor` is not folund in the validation metrics.
    #
    #       * check_finite (bool): When set `True`, stops training when the monitor becomes NaN or infinite.
    #
    #       * stopping_threshold (float): Stop training immediately once the monitored quantity reaches this threshold.
    #
    #       * divergence_threshold (float): Stop training as soon as the monitored quantity becomes worst than this threshold.
    #
    #       * check_on_train_epoch_end (bool): Whether to run early stopping at the time of the training epoch.
    #
    #       * log_rank_zero_only (bool): When `True`, logs the status of the early stopping callback only for rank 0 process.
    #
    #
    #       Operational Usuage:
    #           import FlameModules as flame_modules
    #           from flame_modules.callbacks.early_stopping import EarlyStopping
    #           from flame_modules.trainer.trainer import Trainer
    #
    #           earlyStopping = EarlyStopping(monitor= `val_loss`)
    #           trainer = Trainer(callbacks= [earlyStopping])
    #           
    mode_dict: dict[str, torch.bool] = {
        'min': torch.lt, 'max': torch.gt
    }  
    
    order_dict: dict[str, str] = {
        'min': '<', 'max': '>'
    }
    
    def __init__(self, monitor: str, min_delta: Optional[float] = 0.0, patience: Optional[int] = 3, verbose: Optional[bool] = False, 
                                                                                                    mode: Optional[str] = 'min', 
                                                                                                    strict: Optional[bool] = True, 
                                                                                                    check_finite: Optional[bool] = True, 
                                                                                                    stopping_threshold: Optional[float | None] = None, 
                                                                                                    divergence_threshold: Optional[float | None] = None, 
                                                                                                    check_on_train_epoch_end: Optional[bool | None] = None, 
                                                                                                    log_rank_zero_only: Optional[bool] = False) -> None:

        super(EarlyStopping, self).__init__()
        self.monitor = monitor 
        self.min_delta = min_delta
        self.patience = patience 
        self.verbose = verbose 
        self.mode = model
        self.strict = strict
        self.check_finite = check_finite
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only
        
        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f'Add here ...')
        
        self.min_delta *= 1 if self.monitor_op == torch.lt else -torch_inf
        
        
    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor= self.monitor, mode= self.mode)
    
    
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            #@: If the user runs validation multiple times per training epoch or multiple training epochs without
            #@: validation, then we run after validation instead of `on_train_epoch_end`
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1
            
        
    def _validate_condition_metric(self, logs: dict[str, torch.tensor]) -> bool:
        monitor_val = logs.get(self.monitor)
        error_msg: str = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )
        
        if monitor_val is None:
            if self.strict:
                raise RuntimeErro(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category= RuntimeWarning)
                
            return False
    
        return True
    
    
    @property
    def monitor_op(self) -> Callable[Any]:
        return self.mode_dict[self.mode]
    
    
    def state_dict(self) -> dict[str, Any]:
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }
        
        
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wait_count = state_dict['wait_count']
        self.stopped_epoch = state_dict['stopped_epoch']
        self.best_score = state_dict['best_score']
        self.patience = state_dict['patience']
        
        
    def _should_skip_check(self, trainer: 'flame_modules.Trainer') -> bool:
        from FlameModules.trainer.states import TrainerFn
        return trainer.state.fn != TrainerFn.Fitting or trainer.sanity_checking
    
    
    def on_train_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return 
        self._run_early_stopping_check(trainer)
        
        
    def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        if self._checl_on_train_epoch_end or self._should_skip_check(trainer):
            return 
        self._run_early_stopping_check(trainer)
        
    
    
    def _run_early_stopping_check(self, trainer: 'flame_modules.Trainer') -> None:
        #@: Checks whether the early condition is met and it so tells the trainer to stop the training.
        logs = trainer.callback_metrics
        
        if trainer.fast_dev_run or not self_validate_condition_metric(logs):
            return
        
        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        
        #@: Stop every DDP process if any world process decides to stop.
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all= False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop: 
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
            
        
        
    def _evaluate_stopping_criteria(self, current: torch.tensor) -> tuple[bool, Optional[str]]:
        should_stop: bool = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason: str = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason: str = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason: str = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason: str = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason: str = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )    
        
        return should_stop, reason
    
    

    def _improvement_message(self, current: torch.tensor) -> str:
        #@: Formats a log message that informs the user about an improvement in the monitored score.
        if torch.infinite(self.best_score):
            msg: str = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg: str = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

            
    
    @staticmethod
    def _log_info(trainer: Optional['flame_modules.Trainer'], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy= trainer.strategy if trainer is not None else None
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message: str = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
            

# NOTE : UseCase:    
    #       Operational Usuage:
    #           import FlameModules as flame_modules
    #           from flame_modules.callbacks.early_stopping import EarlyStopping
    #           from flame_modules.trainer.trainer import Trainer
    #
    #           earlyStopping = EarlyStopping(monitor= `val_loss`)
    #           trainer = Trainer(callbacks= [earlyStopping])            
            
