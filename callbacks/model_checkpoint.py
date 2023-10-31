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
import torch, logging, os, re, shutil, time, warnings, yaml
from FlameModules.callbacks.checkpoint import Checkpoint
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from weakref import proxy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)
warning_cache = WarningCache()

# ----------------------------------
# ModelCheckpoint in FlameModules
# ----------------------------------
# What is ModelCheckpoint?
#
#     ModelCheckpoint is a callback utility tailored for FlameModules, a custom deep learning framework. 
#     This utility is designed to monitor a specified metric (like validation loss) during the training 
#     process and persistently save the model's state at specific intervals. This mechanism ensures that 
#     even in long and interrupted training sessions, the most optimal or the most recent state of the 
#     model is always available for future use or reference.
#
# Why do we need ModelCheckpoint in FlameModules?
#
#     1. Capture Optimal Model States: Over the duration of training, a model's performance can vary 
#                                      significantly. The ModelCheckpoint utility guarantees that the version 
#                                      of the model exhibiting the best performance (based on the observed 
#                                      metric) is always saved.
#
#     2. Recovery from Interruptions: Training deep learning models can be a lengthy process. If the 
#                                     training gets interrupted due to any reason like power failures, 
#                                     system crashes, or inadvertent stops, ModelCheckpoint ensures that 
#                                     you don't lose your progress. You can always resume from the last 
#                                     saved state.
#
#     3. Resource Optimization: Training models can be resource-intensive. By utilizing ModelCheckpoint, users can 
#                               refrain from saving the model manually at uncertain intervals. 
#                               Automatic checkpoints save computational time and user effort.
#
#     4. Flexibility in Experimentation: When experimenting with different model architectures, hyperparameters, 
#                                        or data augmentations, ModelCheckpoint provides the flexibility to 
#                                        revert to any saved model state. This makes it easier to compare 
#                                        and evaluate different model versions without retraining them 
#                                        from scratch.
#
#     5. Confidence in Model's Progress: With ModelCheckpoint in place, there's an added confidence in the 
#                                        model's training progression. Users can be assured that regardless 
#                                        of any unforeseen issues, the model's progress is being monitored 
#                                        and captured.
#
# How to use ModelCheckpoint in FlameModules?
#
#     Typically, the ModelCheckpoint is initialized with certain parameters like the path to save, the metric 
#     to monitor, the saving frequency, etc. Once initialized, it can be passed to the FlameModule's training 
#     routine. The callback then automatically handles the saving process based on the provided criteria.

class ModelCheckpoint(Checkpoint):
    # Save the model periodically by monitoring a quantity. Every metric logged with
    # :meth: `flame_modules.FlameModule.log` or :meth: `flame_modules.FlameModule.log_dict` is
    # a candidate for the monitor key. For more info, check `flame_modules.callbacks.checkpoint`
    #
    # After training finished, use :attr: `best_model_path` to retreive the path to the best
    # checkpoint file and :attr: `best_model_score` to retreive its score.
    # 
    # Args:
    #   dirpath : Directory to save the model file.
    #
    #   filename : Checkpoint filename. Can contain named formatting options to be auto-filled.
    #
    #   monitor : Quantity to monitor. By default it is `None` which saves a checkpoint only for the last epoch.
    #
    #   verbose : Verbosity Mode. Default to `None`
    #
    #   save_last : When `True`, saves a `last.ckpt` whenever a checkpoint file gets saved. On a local filesystem,
    #               this will be a symbolic link, and otherwise a copy of the checkpoint file. This allows accessing 
    #               the latest checkpoint in a deterministic manner. Default: `None`.
    #
    #   save_top_k : if `save_top_k == k`,
    #                the best `k` models according to the quantity monitored will be saved. 
    #                if `save_top_k == 0`, no models are saved. 
    #                if `save_top_k == -1', all models are saved.
    #                NOTE :: The monitors are checked every `every_n_epochs` epochs.
    #                if `save_top_k >= 2' and the callback is called multiple times inside
    #                and epoch, the name of the saved file will be appended with a version 
    #                count starting with `v1` unless `enable_version_counter` is set to False.
    #
    #   mode : One of {`min`, `max`}
    #          if `save_top_k != 0`, the decision to overwrite the current save file is made
    #          based on eiter the maimization or the minimization of the monitored quantity.
    #          For `val_acc`, this should be `max`, for `val_loss`, this should be `min`, etc.
    #
    #   auto_insert_metric_name : When `True`, the checkpoints filenames will contain the metric
    #                             name. For Example, `filename='checkpoint_{epoch:02d}-{acc:02.0f}` 
    #                             with epoch `1` and acc `1.12` will resolve to `checkpoint_epoch=01-acc=01.ckpt` 
    #                             It is useful to set it to `False` when metric names contai `/` as this will 
    #                             result in extra folders.
    #
    #   save_weights_only : If `True`, then only the model's weights will be saved. Otherwise, the optimizer 
    #                       states, lr-scheduler states, etc are added in the checkpoint too.
    #
    #   every_n_train_steps : Number of training steps between checkpoints.
    #                         If `every_n_train_steps == None or every_n_train_steps == 0`, we skip saving during training.
    #                         To disable, set `every_n_train_steps = 0`. This value must be `None` or Non-Negative.
    #                         This must be mutually exclusive with `train_time_interval` and `every_n_epochs`.
    #
    #   train_time_interval : Checkpoints are monitored at the specified time interval. 
    #                         For all practical purposes, this cannot be smaller than the amount of time it takes to process
    #                         a single training batch. This is not guaranteed to execute at the exact time specified, but should
    #                         be close. This must be mutually exclusive with `every_n_train_steps` and `every_n_epochs`.
    #
    #   every_n_epochs : Number of epochs between checkpoints.
    #                    This value must be `None` or `Non-Negative`. To disable saving top-k checkpoints, set `every_n_epochs = 0`.
    #                    This argument does not impact the saving of `save_last = True` checkpoints. 
    #
    #   save_on_train_epoch_end : Whether to run checkpointing at the end of the training epoch. If this is `False`, then the check
    #                             runs at the end of the validation. 
    #
    #   enable_version_counter : Whether to append a version to the existing file name. If this is `False`, then the checkpoint files 
    #                            will be overwritten. 
    #
    #       Code Docs:
    #           import FlameModules as flame_modules
    #           from flame_modules.trainer.trainer import Trainer
    #           from flame_modules.callbacks.model_checkpoint import ModelCheckpoint
    #
    #           checkpoint_callback = ModelCheckpoint(dirpath= ...)
    #           trainer = Trainer(callbacks= [checkpoint_callback])
    #
    #           or 
    #           checkpoint_callback_2 = ModelCheckpoint(monitor= 'val_loss', dirpath= ..., filename= 'sample_dataset-{epoch:02d}-{val_loss:.2f}')
    #           trainer = Trainer(callbacks= [checkpoint_callback])
    #           #@: saves a file like: my/path.sample_dataset-epoch=02-val_loss=0.32.ckpt
    
    
    checkpoint_join_char: str = '-'
    checkpoint_equals_char: str = '='
    checkpoint_name_last: str = 'last'
    file_extension: str = '.ckpt'
    starting_version: int = 1
    
    def __init__(self, dirpath: Optional[str] = None, filename: Optional[str] = None, monitor: Optional[str] = None, 
                                                                                      verbose: Optional[bool] = False,
                                                                                      save_last: Optional[bool] = None, 
                                                                                      save_top_k: Optional[int] = 1, 
                                                                                      save_weights_only: Optional[bool] = False,
                                                                                      mode: Optional[str] = 'min',
                                                                                      auto_insert_metric_name: Optional[bool] = True,
                                                                                      every_n_train_steps: Optional[int] = None,
                                                                                      train_time_interval: Optional[timedelta] = None, 
                                                                                      every_n_epochs: Optional[int] = None,
                                                                                      save_on_train_epoch_end: Optional[bool] = None,
                                                                                      enable_version_counter: Optional[bool] = True) -> None:

        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter
        self._last_global_step_saved = 0
        self._last_time_checked: Optional[float] = None
        self.current_score: Optional[torch.tensor] = None
        self.best_k_models: dict[str, torch.tensor] = {}
        self.kth_best_model_path: str = ''
        self.best_model_score: Optional[torch.tensor] = None
        self.best_model_path: str = ''
        self.last_model_path: str = ''
        self._last_checkpoint_saved: str = ''
        
        self.kth_value: torch.tensor
        self.dirpath: Optional[str]
        self.__init_monitor_mode(mode)
        self.__init_ckpt_dir(dirpath, filename)
        self.__init_triggers(every_n_train_steps, every_n_epochs, train_time_interval)
        self.__validate_init_configuration()
        
        

    # Understanding the `state_key` property in the ModelCheckpoint class of FlameModules
    # ------------------------------------------------------------------------------------
    # What is the `state_key` property?
    #
    #     The `state_key` property is a dynamic attribute in the ModelCheckpoint class that, when accessed, 
    #     returns a string representation generated by the `_generate_state_key` method. This key uniquely 
    #     represents a checkpoint's state based on the various parameters of the ModelCheckpoint.
    #
    # How does `state_key` work?
    #
    #     1. The `@property` decorator: This decorator transforms the method into a property-like attribute 
    #                                   of the class. This means when you access `state_key`, you are essentially 
    #                                   calling the `state_key` method but without the parentheses.
    #
    #     2. Calling `_generate_state_key`: Inside the property, the method `_generate_state_key` is invoked 
    #                                       with various parameters from the class instance. This method is 
    #                                       responsible for generating the unique key string based on the 
    #                                       passed parameters.
    #
    # Parameters used to generate the state key:
    #
    #     1. `monitor`: The metric that the ModelCheckpoint is monitoring. For example, it could be 'val_loss' 
    #                   indicating that the model's validation loss is being tracked.
    #
    #     2. `mode`: Represents the optimization mode, which could either be 'max' for metrics where higher 
    #                 values are better (like accuracy) or 'min' for metrics where lower values are preferred 
    #                 (like loss).
    #
    #     3. `every_n_train_steps`: This parameter specifies the frequency, in terms of training steps, at which 
    #                               the model should be saved.
    #
    #     4. `every_n_epochs`: This indicates the frequency, in terms of epochs, for saving the model.
    #
    #     5. `train_time_interval`: Represents the time interval for saving the model based on the training duration.
    #
    # Why is the `state_key` property useful?
    #
    #     Generating a unique key for each checkpoint state ensures that each saved model can be distinctly 
    #     identified and retrieved. This becomes particularly valuable when dealing with numerous checkpoints 
    #     and needing to quickly distinguish between them based on their training characteristics.
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor= self.monitor,
            mode= self.mode,
            every_n_train_steps= self._every_n_train_steps, 
            every_n_epochs= self._every_n_epochs,
            train_time_interval= self._train_time_interval
        )
        
        
        
    # `setup` Method in the ModelCheckpoint Class of FlameModules
    # ------------------------------------------------------------
    # Purpose:
    #     The `setup` method initializes the necessary configurations related to checkpointing before training starts.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, responsible for training management.
    #   - `flame_module`: Represents the model being trained, an instance of `flame_modules.FlameModule`.
    #   - `stage`: Current stage of training, usually 'fit'.
    #
    # Key Steps:
    #   1. Resolve the checkpoint directory path and synchronize it across training strategies.
    #   2. Establish the filesystem (`_fs`) based on the resolved directory path.
    #   3. If the current process is the primary one (`global_zero`) and in 'fit' stage, it checks and warns if the directory already has content.
        
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        dirpath: str = self.__resolve_ckpt_dir(trainer)
        dirpath: str = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        self._fs = get_filesystem(self.dirpath or '')
        if trainer.is_global_zero and stage == 'fit':
            self.__warn_if_dir_not_empty(self.dirpath)
            
            
            
            
            
    # `on_train_start` Method in ModelCheckpoint Class
    # -------------------------------------------------
    # Purpose:
    #     Initializes checkpoint timings when training begins.
    #
    # Parameters:
    # - `trainer`: Instance of `flame_modules.Trainer`.
    # - `flame_module`: Model instance, `flame_modules.FlameModule`.
    #
    # Key Step:
    # 1. Capture the current monotonic time as `_last_time_checked` to track elapsed time during training.
    def on_train_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self._last_time_checked = time.monotonic()
        
        
        
        
    # `on_train_batch_end` Method in ModelCheckpoint Class
    # -----------------------------------------------------
    # Purpose:
    #     Determines if a checkpoint should be saved after each training batch ends based on specific criteria.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
    #   - `outputs`: The output from the training step.
    #   - `batch`: The current training batch.
    #   - `batch_index`: The index of the current training batch.
    #
    # Key Steps:
    #   1. Check if checkpoint saving should be skipped.
    #   2. Determine if checkpointing should be skipped based on training steps or elapsed time.
    #   3. If checkpoints are based on time intervals, check and sync the decision across different ranks.
    #   4. Retrieve available monitor metrics.
    #   5. Save the top-k checkpoints.
    #   6. Save the most recent checkpoint.

    def on_train_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_output', batch: Any, batch_index: int = int) -> None:
        #@: Saves checkpoint on train batch end if we meet the criteria for `every_n_train_steps`
        if self._should_skip_saving_checkpoint(trainer):
            return 
        
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)
        
        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            #@: In case we have time differences across rank
            #@: broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)
            
        if skip_batch and skip_time:
            return 
        if not skip_time:
            self._last_time_checked = now
            
        
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)
        
        
        
        
        
    # `on_train_epoch_end` Method in ModelCheckpoint Class
    # -----------------------------------------------------
    # Purpose:
    #     Determines if a checkpoint should be saved at the end of each training epoch based on specific criteria.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
    #
    # Key Steps:
    #   1. Determine if checkpoint saving should be skipped.
    #   2. Check if a checkpoint should be saved at this specific epoch end.
    #   3. Retrieve available monitor metrics.
    #   4. Save the top-k checkpoints if epoch conditions are met.
    #   5. Save the most recent checkpoint.
    
    def on_train_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Save a checkpoint at the end of the training epoch
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                
            self._save_last_checkpoint(trainer, monitor_candidates)
            
            
            
    
    
    # `on_validation_end` Method in ModelCheckpoint Class
    # -----------------------------------------------------
    # Purpose:
    #     Determines if a checkpoint should be saved at the end of the validation stage based on specific criteria.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
    #
    # Key Steps:
    #   1. Determine if checkpoint saving should be skipped.
    #   2. Check that a checkpoint shouldn't be saved at this specific epoch end (opposite of training condition).
    #   3. Retrieve available monitor metrics.
    #   4. Save the top-k checkpoints if epoch conditions are met.
    #   5. Save the most recent checkpoint.

    def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Save a checkpoint at the end of the validation stage.
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            
            self._save_last_checkpoint(trainer, monitor_candidates)
            
            
    
    
    
    
    # `state_dict` Method in ModelCheckpoint Class
    # -------------------------------------------
    # Purpose:
    #     Retrieves the current state of the checkpoint instance as a dictionary. Useful for serialization or logging.
    #
    # Returns:
    #   - A dictionary containing key attributes of the ModelCheckpoint instance.
    #
    # Key Components:
    #   1. `monitor`: The metric being monitored.
    #   2. `best_model_score`: Score of the best model checkpoint.
    #   3. `best_model_path`: File path to the best model checkpoint.
    #   4. `current_score`: Score of the current model.
    #   5. `dirpath`: Directory where checkpoints are stored.
    #   6. `best_k_models`: Dictionary of best 'k' model paths with their corresponding scores.
    #   7. `kth_best_model_path`: File path to the k-th best model checkpoint.
    #   8. `kth_value`: The score of the k-th best model.
    #   9. `last_model_path`: File path to the most recent checkpoint.

    def state_dict(self) -> dict[str, Any]:
        return {
            'monitor': self.monitor,
            'best_model_score': self.best_model_score,
            'best_model_path': self.best_model_path,
            'current_score': self.current_score,
            'dirpath': self.dirpath,
            'best_k_models': self.best_k_models,
            'kth_best_model_path': self.kth_best_model_path,
            'kth_value': self.kth_value,
            'last_model_path': self.last_model_path
        }
        
        
        
        
    # `load_state_dict` Method in ModelCheckpoint Class
    # -------------------------------------------------
    # Purpose:
    #     Loads the state of the ModelCheckpoint instance from a given dictionary. Used to restore the checkpoint state.
    #
    # Parameters:
    #   - `state_dict`: A dictionary containing key attributes of the ModelCheckpoint instance.
    #
    # Key Steps:
    #   1. Check if the directory path matches the stored path.
    #   2. If it matches, update the instance attributes with the values from `state_dict`.
    #   3. If the directory path does not match, warn the user and only update the `best_model_path` attribute.
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        dirpath_from_ckpt = state_dict.get('dirpath', self.dirpath)
        
        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict['best_model_score']
            self.kth_best_model_path = state_dict.get('kth_best_model_path', self.kth_best_model_path)
            self.kth_value = state_dict.get('kth_value', self.kth_value)
            self.best_k_models = state_dict.get('best_k_models', self.best_k_models)
            self.last_model_path = state_dict.get('last_model_path', self.last_model_path)
            
        else:
            warnings.warn(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."   
            )
            
        self.best_model_path = state_dict['best_model_path']
        
        
    
    
    
    
    # `_save_topk_checkpoint` Method in ModelCheckpoint Class
    # -------------------------------------------------------
    # Purpose:
    #     Saves the top-k checkpoints based on the specified monitoring metric.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `monitor_candidates`: Dictionary containing metrics to monitor.
    #
    # Key Steps:
    #   1. If `save_top_k` is set to 0, no checkpoints are saved.
    #   2. If a monitor metric is specified, validate if it's present in the metrics returned.
    #   3. If the metric is not found and validation has run, raise a misconfiguration error.
    #   4. Depending on whether a monitor is specified, save the checkpoint based on that metric or save without monitoring any specific metric.

    def _save_topk_checkpoint(self, trainer: 'flame_modules.Trainer', monitor_candidates: dict[str, torch.tensor]) -> None:
        if self.save_top_k == 0:
            return
        
        #@: Validate metric 
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                m: str = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                )
                if trainer.fit_loop.epoch_loop.val_loop._has_run:
                    raise MisconfigurationException(m)
                warning_cache.warn(m)
            self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)
            
            
    
    
    
    # `_save_checkpoint` Method in ModelCheckpoint Class
    # --------------------------------------------------
    # Purpose:
    #     Saves the checkpoint of the model and notifies loggers.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `filepath`: Path where the checkpoint will be saved.
    #
    # Key Steps:
    #   1. Use the trainer to save the checkpoint at the specified filepath. Decide to save weights only based on `self.save_weights_only`.
    #   2. Update the last saved global step and filepath information.
    #   3. If the current step is a global step, notify all the loggers that a checkpoint has been saved.
    
    def _save_checkpoint(self, trainer: 'flame_modules.Trainer', filepath: str) -> None:
        trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath
        
        #@: Notify the loggers
        if trainer.is_global_step:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
                
    
    
    
    
    
    
    # `_link_checkpoint` Static Method in ModelCheckpoint Class
    # --------------------------------------------------------
    # Purpose:
    #     Create a symbolic link to the saved checkpoint file.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #   - `filepath`: Path where the original checkpoint is saved.
    #   - `linkpath`: Path where the symbolic link will be created.
    #
    # Key Steps:
    #   1. If the current step is a global step, check if a link or file already exists at `linkpath`.
    #   2. Remove the existing link or file at `linkpath`.
    #   3. Create a symbolic link from `filepath` to `linkpath`.
    #   4. Initiate a barrier synchronization among processes in the distributed training environment.

    @staticmethod
    def _link_checkpoint(trainer: 'flame_modules.Trainer', filepath: str, linkpath: str) -> None:
        if trainer.is_global_step:
            if os.path.islink(linkpath) or os.path.isfile(linkpath):
                os.remove(linkpath)
            elif os.path.isdir(linkpath):
                shutil.rmtree(linkpath)
            os.symlink(filepath, linkpath)
        trainer.strategy.barrier()
        
        
        
        
    # `_should_skip_saving_checkpoint` Method in ModelCheckpoint Class
    # ---------------------------------------------------------------
    # Purpose:
    #     Determine if a checkpoint should be skipped based on various training conditions.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #
    # Returns:
    #   - A boolean value indicating whether to skip checkpoint saving.
    #
    # Key Conditions:
    #   1. Skip if the trainer is in `fast_dev_run` mode.
    #   2. Skip if the trainer's current function isn't for fitting.
    #   3. Skip during sanity checks.
    #   4. Skip if a checkpoint was already saved in the last training step.

    def _should_skip_saving_checkpoint(self, trainer: 'flame_modules.Trainer') -> bool:
        from FlameModules.trainer.states import TrainerFn
        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.Fitting  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )
        
    
    
    
    
    
    # `_should_save_on_train_epoch_end` Method in ModelCheckpoint Class
    # ----------------------------------------------------------------
    # Purpose:
    #     Determine if a checkpoint should be saved at the end of a training epoch based on specific conditions.
    #
    # Parameters:
    #   - `trainer`: Instance of `flame_modules.Trainer`.
    #
    # Returns:
    #   - A boolean value indicating whether to save checkpoint at the end of the training epoch.
    #
    # Key Conditions:
    #   1. Directly return the predefined `_save_on_train_epoch_end` value if it's set.
    #   2. If validation checks don't occur every epoch, don't save.
    #   3. Save if there are no validation batches.
    #   4. If multiple validations run within a training epoch, save only after validation rather than at epoch end.

    def _should_save_on_train_epoch_end(self, trainer: 'flame_modules.Trainer') -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end
        
        #@: If `check_val_every_n_epoch != 1`, we cannot say when the validation dataloader will be loaded
        if trainer.check_val_every_n_epoch != 1:
            return False
        
        #@: No Validation means save on train epoch end
        num_val_batches: tuple[Any] = (
            sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
        )
        if num_val_batches == 0:
            return True
        
        #@: If the use runs validation multiple times per training epoch, then we run after validation
        #@: instead of on train epoch end.
        
        return trainer.val_check_interval == 1.0
    
    
    
    
    
    # `__validate_init_configuration` Method in ModelCheckpoint Class
    # --------------------------------------------------------------
    # Purpose:
    #     Validate the configuration settings upon the initialization of the ModelCheckpoint class.
    #     Ensures that the settings provided to the class make logical sense and are compatible with one another.
    #
    # Key Steps:
    #   1. Validate the value of `save_top_k`. It should be greater than or equal to -1.
    #   2. Validate the value of `_every_n_train_steps`. It should be non-negative.
    #   3. Validate the value of `_every_n_epochs`. It should be non-negative.
    #   4. Ensure that only one of `_every_n_train_steps`, `_every_n_epochs`, or `_train_time_interval` is set.
    #      They should be mutually exclusive to avoid ambiguity.
    #   5. If the `monitor` attribute is None and `save_top_k` isn't -1, 0, or 1, raise an error.
    #      (Since there's no specified metric for top_k to track.)
    #
    # Throws:
    #   - MisconfigurationException if any of the checks fail.

    def __validate_init_configuration(self) -> None:
        if self.save_top_k < -1:
            raise MisconfigurationException(f'Invalid value for save_top_k={self.save_top_k}. Must be >= -1')
        
        if self._every_n_train_steps < 0:
            raise MisconfigurationException(
                f'Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0'
            )
        
        if self._every_n_epochs < 0:
            raise MisconfigurationException(f'Invalid value for every_n_epochs={self._every_n_epochs}. Must be >= 0')
          
        every_n_train_steps_triggered = self._every_n_train_steps >= 1
        every_n_epochs_triggered = self._every_n_epochs >= 1
        train_time_interval_triggered = self._train_time_interval is not None
        if every_n_train_steps_triggered + every_n_epochs_triggered + train_time_interval_triggered > 1:
            raise MisconfigurationException(
                f"Combination of parameters every_n_train_steps={self._every_n_train_steps}, "
                f"every_n_epochs={self._every_n_epochs} and train_time_interval={self._train_time_interval} "
                "should be mutually exclusive."
            )
            
        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            #@: -1: save all epochs, 0: nothing is saved, 1: save last epoch
            raise MisconfigurationException(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid"
                " configuration. No quantity for top_k to track."
            )
    
    
    
    
    # `__init_ckpt_dir` Method in ModelCheckpoint Class
    # -------------------------------------------------
    # Purpose:
    #     Initialize the directory path and filename for checkpoint saving.
    #     Ensures that the directory is accessible and checks the underlying filesystem protocol.
    #
    # Parameters:
    #   - `dirpath`: The directory path where the checkpoint will be saved. Can be None.
    #   - `filename`: The name of the checkpoint file.
    #
    # Key Steps:
    #   1. Determine the filesystem of the provided directory using `get_filesystem` method.
    #   2. If a directory path is provided and its filesystem protocol is 'file', resolve the path to its real location.
    #   3. Assign the resolved directory path and filename to the object's attributes.

    def __init_ckpt_dir(self, dirpath: Optional[str], filename: Optional[str]) -> None:
        self._fs = get_filesystem(dirpath if dirpath else '')
        if dirpath and self._fs.protocol == 'file':
            dirpath = os.path.realpath(dirpath)
            
        self.dirpath = dirpath
        self.filename = filename
        
    
    
    
    # `__init_monitor_mode` Method in ModelCheckpoint Class
    # ----------------------------------------------------
    # Purpose:
    #     Initialize the monitor mode for the checkpoint. The monitor mode determines
    #     whether the checkpoint should track the minimum or maximum of a given metric.
    #
    # Parameters:
    #   - `mode`: The desired mode for monitoring which can be 'min' or 'max'.
    #
    # Key Steps:
    #   1. Define infinite values using torch.tensor.
    #   2. Construct a dictionary `mode_dict` that maps the mode to its corresponding
    #      initialization value and the mode itself.
    #   3. Check if the provided `mode` exists in `mode_dict`. If not, raise a MisconfigurationException.
    #   4. Assign the values from `mode_dict` based on the provided mode to the object's attributes.

    def __init_monitor_mode(self, mode: str) -> None:
        torch_inf = torch.tensor(torch.inf)
        mode_dict: dict[str, tuple[torch.tensor, str]] = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max')
        }
        
        if mode not in mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}")
        
        self.kth_value, self.mode = mode_dict[mode]
        
        
        
        
    
    # `__init_triggers` Method in ModelCheckpoint Class
    # --------------------------------------------------
    # Purpose:
    #     Initializes the triggers for when to save a checkpoint based on training steps, epochs, or a time interval.
    #
    # Parameters:
    #   - `every_n_train_steps`: Frequency (in training steps) at which checkpoints are saved.
    #   - `every_n_epochs`: Frequency (in epochs) at which checkpoints are saved.
    #   - `train_time_interval`: Time interval after which checkpoints are saved.
    #
    # Key Steps:
    #   1. If none of the triggers (`every_n_train_steps`, `every_n_epochs`, and `train_time_interval`) are set, default 
    #      to saving once after each validation epoch. Set `every_n_epochs` to 1 and `every_n_train_steps` to 0.
    #   2. If one or more triggers are set, use the provided values or default to 0 if they are None.
    #   3. Assign the values to the object's attributes.

    def __init_triggers(self, every_n_train_steps: Optional[int], every_n_epochs: Optional[int], train_time_interval: Optional[timedelta]) -> None:
        #@: Default to running once after each validation epoch if neither
        #@: `every_n_train_steps` not `every_n_epochs` is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            log.debug('Both `every_n_train_steps` and `every_n_epochs` are not set. Setting `every_n_epochs 1`')
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0
            
        self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps
        
        
        
    @property
    def every_n_epochs(self) -> Optional[int]:
        return self._every_n_epochs
    
    
    
    
    
    
    
    # `check_monitor_top_k` Method in ModelCheckpoint Class
    # ----------------------------------------------------
    # Purpose:
    #     Determines if the current checkpoint should be saved based on the ranking of the monitored quantity.
    #
    # Parameters:
    #   - `trainer`: Instance of the training framework, often used to access training state and utilities.
    #   - `current`: The current value of the monitored quantity.
    #
    # Key Steps:
    #   1. If no current value is provided, return `False` since there's nothing to compare against.
    #   2. If `save_top_k` is set to -1, save all checkpoints and return `True`.
    #   3. If the number of currently saved models (`best_k_models`) is less than the specified `save_top_k`, return `True`.
    #   4. Determine the operation for comparison based on the mode (either `min` or `max`).
    #   5. Check if the current checkpoint should update the best models list and save.
    #   6. If multi-device training (e.g., distributed training), gather the decision across all devices and return the reduced result.

    def check_monitor_top_k(self, trainer: 'flame_modules.Trainer', current: Optional[torch.tensor] = None) -> bool:
        if current is None:
            return False
        
        if self.save_top_k == -1:
            return True
        
        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True
        
        monitor_op = {
            'min': torch.lt, 
            'max': torch.gt
        }[self.mode]
        
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))
        return should_update_best_and_save
    
    
    
    
    
    
    # `_format_checkpoint_name` Class Method in ModelCheckpoint Class
    # ---------------------------------------------------------------
    # Purpose:
    #     Formats the checkpoint filename based on user-defined patterns, metrics, and certain conditions.
    #
    # Parameters:
    #   - `filename`: Template for the filename which might contain placeholders for metrics.
    #   - `metrics`: Dictionary containing metric names and their respective values.
    #   - `prefix`: Prefix to be added before the filename.
    #   - `auto_insert_metric_name`: If `True`, metric names are automatically inserted into the filename.
    #
    # Key Steps:
    #   1. If the user hasn't provided a filename, use a default format based on epoch and step.
    #   2. Extract all placeholders from the filename using regular expressions.
    #   3. Sort the extracted placeholders from longest to shortest to ensure correct substitution.
    #   4. Loop through each placeholder:
    #       a. If `auto_insert_metric_name` is enabled, update the filename by inserting the metric name before its value.
    #       b. Replace the placeholder in the filename with its respective metric value from the `metrics` dictionary.
    #       c. If a placeholder doesn't have a corresponding value in the metrics, default its value to 0.
    #   5. If a prefix is provided, prepend it to the filename.
    #   6. Return the formatted filename.

    @classmethod
    def _format_checkpoint_name(cls, filename: Optional[str], metrics: dict[str, torch.tensor], prefix: str = '', auto_insert_metric_name: bool = True) -> str:
        if not filename:
            #@: filename is not set, therefore, use default name
            filename: str = '{epoch}' + cls.checkpoint_join_char + '{step}'
            
        #@: Check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        
        #@: Sort keys from the longest to shortest to avoid replacing substring
        #@: Eg: if keys are `epoch` and `epoch_test`, the latter must be replaced first
        groups = sorted(groups, key= lambda x: len(x), reverse= True)
        
        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + cls.checkpoint_equals_char + '{' + name)                
            filename = filename.replace(group, f"{{0[{name}]")
            
            if name not in metrics:
                metrics[name] = torch.tensor(0)
        
        filename = filename.format(metrics)
        if prefix:
            filename = cls.checkpoint_join_char.join([prefix, filename])
        
        return filename
    
    
    
    
    
    # `format_checkpoint_name` Method in ModelCheckpoint Class
    # --------------------------------------------------------
    # Purpose:
    #     Given metrics and optional parameters, it constructs the full checkpoint filename along with path, 
    #     if the directory path (`dirpath`) is provided.
    #
    # Parameters:
    #   - `metrics`: Dictionary containing metric names and their respective values.
    #   - `filename`: Template for the filename which might contain placeholders for metrics. Defaults to class's filename attribute.
    #   - `ver`: Version number which will be added as a suffix to the filename.
    #
    # Key Steps:
    #   1. If a filename isn't provided, use the default filename from the class's attribute.
    #   2. Format the filename using the `_format_checkpoint_name` method, which replaces placeholders in the filename with their respective metric values.
    #   3. If a version number is provided, append it to the filename.
    #   4. Construct the full checkpoint name by adding the file extension.
    #   5. If a directory path (`dirpath`) is provided, prepend it to the checkpoint name to form the full path.
    #   6. Return the full path or checkpoint name, as per the conditions.

    def format_checkpoint_name(self, metrics: dict[str, torch.tensor], filename: Optional[str] = None, ver: Optional[int] = None) -> str:
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename= filename,
            metrics= metrics,
            auto_insert_metric_name= self.auto_insert_metric_name
        )
        
        if ver is not None:
            filename = self.checkpoint_join_char.join((filename, f'v{ver}'))
            
        ckpt_name = f'{filename}{self.file_extension}'
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name
    
    
    
    
    
    
    # `__resolve_ckpt_dir` Method in ModelCheckpoint Class
    # ----------------------------------------------------
    # Purpose:
    #     Determines the appropriate directory path for saving model checkpoints based on certain conditions.
    #     It uses attributes from both the ModelCheckpoint class and the Trainer's logger to identify the 
    #     optimal path to save the checkpoints.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` which provides necessary attributes to determine the save directory.
    #
    # Key Steps:
    #   1. If the `ModelCheckpoint` class has its `dirpath` attribute set, then use this as the save directory.
    #   2. If the trainer has loggers, use the `loggers` attributes to determine the save directory. If the first logger has 
    #      a `save_dir` attribute set, use that. If not, use the trainer's `default_root_dir`. The final directory 
    #      is then constructed by joining the `save_dir`, logger's `name`, logger's `version`, and "checkpoints".
    #   3. If there are no loggers, use the `default_root_dir` of the trainer and append "checkpoints" to it.
    #   4. Return the constructed checkpoint directory path.

    def __resolve_ckpt_dir(self, trainer: 'flame_modules.Trainer') -> _PATH:
        #@: Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        #@: determine where to save checkpoints. The path for saving weights is set in this priority:
        #       1. The `ModelCheckpoint`'s `dirpath` if passed in
        #       2. The `Logger`'s `log_dir` if the trainer has loggers
        #       3. The `Trainer`'s `default_root_dir` if the trainer has no loggers

        # The path gets extended with subdirectory "checkpoints".

        if self.dirpath is not None:
            #@: Short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath
        
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f'version_{version}'
            ckpt_path = os.path.join(save_dir, str(name), version, 'checkpoints')
        else:
            #@: If no loggers, use the `default_root_dir`
            ckpt_path = os.path.join(trainer.default_root_dir, 'checkpoints')
        
        return ckpt_path
    
    
    
    
    
    # `_find_last_checkpoints` Method in ModelCheckpoint Class
    # ---------------------------------------------------------
    # Purpose:
    #     Scans the determined checkpoint directory to find all checkpoint files that match the naming pattern 
    #     for "last" checkpoints. It then returns a set of these checkpoint file paths.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_models.Trainer` which is used to determine the checkpoint save directory.
    #
    # Key Steps:
    #   1. Resolve the checkpoint directory path by calling the `__resolve_ckpt_dir` method.
    #   2. Check if the resolved directory path exists.
    #   3. If the directory exists, list all the files within it.
    #   4. Filter the list of files to include only those that contain the naming pattern for "last" checkpoints 
    #      as determined by the `checkpoint_name_last` attribute.
    #   5. Normalize the paths of these files and return them as a set.

    def _find_last_checkpoints(self, trainer: 'flame_models.Trainer') -> set[str]:
        #@: Find all checkpoints in the folder
        ckpt_path = self.__resolve_ckpt_dir(trainer)
        if self._fs.exists(ckpt_path):
            return {
                os.path.normpath() 
                for p in self._fs.ls(ckpt_path, detail= False)
                if self.checkpoint_name_last in os.path.split(p)[1]
            }
        return set()
    
    
    
    
    
    # `__warn_if_dir_not_empty` Method in ModelCheckpoint Class
    # --------------------------------------------------------
    # Purpose:
    #     Checks if the checkpoint directory is not empty and issues a warning if it's not empty. 
    #     This is used to alert users that they might be overwriting or mixing checkpoints from previous runs.
    #
    # Parameters:
    #   - `dirpath`: The directory path (of type _PATH or str) where checkpoints are saved or will be saved.
    #
    # Key Steps:
    #   1. Check if the `save_top_k` attribute is not equal to 0. 
    #      (This implies that there are checkpoints that will be saved or have been saved.)
    #   2. Use the `_is_dir` function to check if the `dirpath` is indeed a directory.
    #   3. If the directory exists, list all the files within it using the `_fs.ls` method.
    #   4. If there are any files in the directory, issue a warning to inform the user 
    #      that the checkpoint directory is not empty.

    def __warn_if_dir_not_empty(self, dirpath: _PATH | str) -> None:
        if self.save_top_k != 0 and _is_dir(self._fs, dirpath, strict= True) and len(self._fs.ls(dirpath)) > 0:
            rank_zero_warn(f'Checkpoint directory {dirpath} exists and is not empty')
            
            
        
        
        
    # `_get_metric_interpolated_filepath_name` Method in ModelCheckpoint Class
    # ----------------------------------------------------------------------
    # Purpose:
    #     Constructs the checkpoint file path based on metrics, and ensures 
    #     uniqueness of the file name by appending a version count if necessary.
    #
    # Parameters:
    #   - `monitor_candidates`: A dictionary containing metric names as keys 
    #                           and their corresponding tensor values.
    #   - `trainer`: An instance of `flame_modules.Trainer` used for checking 
    #                the existence of files.
    #   - `del_filepath`: An optional file path to be ignored when checking 
    #                     for file existence. This is useful when the file 
    #                     is to be overwritten or deleted in the next steps.
    #
    # Key Steps:
    #   1. Format the checkpoint file name using the `format_checkpoint_name` method.
    #   2. If version counting is enabled:
    #      a. Set the initial version count using `starting_version`.
    #      b. Loop until a file path is found that doesn't already exist (and is not the one to be deleted).
    #         - For each iteration, increment the version count and update the file path.
    #   3. Return the unique file path.

    def _get_metric_interpolated_filepath_name(self, monitor_candidates: dict[str, torch.tensor], trainer: 'flame_modules.Trainer',
                                                                                                  del_filepath: Optional[str] = None) -> str:
        
        filepath = self.format_checkpoint_name(monitor_candidates)
        if self._enable_version_counter:
            version_cnt: int = self.starting_version
            while self.file_exists(filepath, trainer) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(monitor_candidates, ver= version_cnt)
                version_cnt += 1
        
        return filepath
    
    
    
    
    # `_monitor_candidates` Method in ModelCheckpoint Class
    # -----------------------------------------------------
    # Purpose:
    #     Generates a dictionary of metric candidates for monitoring, including 
    #     the current epoch and step, based on the trainer's callback metrics.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` which provides 
    #                callback metrics and details about the current training 
    #                state.
    #
    # Key Steps:
    #   1. Deepcopy the trainer's callback metrics to avoid modifying the original.
    #   2. Check and set the 'epoch' in monitor candidates:
    #      a. If 'epoch' is a tensor, convert it to an integer.
    #      b. If 'epoch' is not present in callback metrics, use `trainer.current_epoch`.
    #   3. Check and set the 'step' in monitor candidates:
    #      a. If 'step' is a tensor, convert it to an integer.
    #      b. If 'step' is not present in callback metrics, use `trainer.global_step`.
    #   4. Return the updated monitor candidates dictionary.

    def _monitor_candidates(self, trainer: 'flame_modules.Trainer') -> dict[str, torch.tensor]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        epoch = monitor_candidates.get('epoch')
        monitor_candidates['epoch'] = epoch.int() if isinstance(epoch, torch.tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get('step')
        monitor_candidates['step'] = step.int() if isinstance(step, torch.tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates
    
    
    
    
    # `_save_last_checkpoint` Method in ModelCheckpoint Class
    # -------------------------------------------------------
    # Purpose:
    #     This method handles the saving of the last checkpoint during training.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` that provides 
    #                details about the current training state.
    #   - `monitor_candidates`: A dictionary containing metrics used for 
    #                           checkpoint naming interpolation.
    #
    # Key Steps:
    #   1. If `save_last` attribute is set to False, the method returns without performing any action.
    #   2. Format the checkpoint name using monitor candidates and the attribute `checkpoint_name_last`.
    #   3. If version counting is enabled:
    #      a. Initialize the version counter with `starting_version`.
    #      b. If a file exists with the given name and it's not the last model path, iterate to find an unused name.
    #   4. Set the `last_model_path` to the determined filepath.
    #   5. If the file system protocol is 'file', the last checkpoint was saved, and the `save_top_k` attribute isn't 0:
    #      a. Link the last saved checkpoint to the new file path.
    #   6. Otherwise, save the checkpoint.
    #   7. If a previous checkpoint exists and should be removed, remove it.

    def _save_last_checkpoint(self, trainer: 'flame_modules.Trainer', monitor_candidates: dict[str, torch.tensor]) -> None:
        if not self.save_last:
            return 
        
        filepath = self.format_checkpoint_name(monitor_candidates, self.checkpoint_name_last)
        
        if self._enable_version_counter:
            version_cnt: int = self.starting_version
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.checkpoint_name_last, ver= version_cnt)
                version_cnt += 1
                
        previous, self.last_model_path = self.last_model_path, filepath
        if self._fs.protocol == 'file' and self._last_checkpoint_saved and self.save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, previous)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)
            
        
    
    
    
    # `_save_monitor_checkpoint` Method in ModelCheckpoint Class
    # ---------------------------------------------------------
    # Purpose:
    #     This method saves a checkpoint based on the monitored metric value, ensuring only the top-k checkpoints 
    #     (defined by `save_top_k`) are retained.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` that provides 
    #                details about the current training state.
    #   - `monitor_candidates`: A dictionary containing metrics used for checkpoint saving decision.
    #
    # Key Steps:
    #   1. Assert that the `monitor` attribute (metric name) is set.
    #   2. Get the current value of the monitored metric from `monitor_candidates`.
    #   3. If the current metric value qualifies for top-k:
    #      a. Assert that the current metric value is not None.
    #      b. Update the best model details and save the checkpoint.
    #   4. If `verbose` attribute is True and the metric value didn't qualify for top-k:
    #      a. Display an informational message indicating that the current checkpoint isn't in the top-k.

    def _save_monitor_checkpoint(self, trainer: 'flame_modules.Trainer', monitor_candidates: dict[str, torch.tensor]) -> None:
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            epoch = monitor_candidates['epoch']
            step = monitor_candidates['step']
            rank_zero_info(f'Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}')
            
            
            
    
    # `_save_none_monitor_checkpoint` Method in ModelCheckpoint Class
    # --------------------------------------------------------------
    # Purpose:
    #     This method saves a checkpoint when there's no specific metric set to monitor. 
    #     It ensures that if `save_top_k` is set to 1, only the most recent checkpoint is retained.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` that provides 
    #                details about the current training state.
    #   - `monitor_candidates`: A dictionary containing metrics which can be used in the file name.
    #
    # Key Steps:
    #   1. Get the interpolated filepath name for the checkpoint, based on current metrics and existing file structure.
    #   2. Set the `best_model_path` to the newly computed `filepath` (retaining the old path in `previous`).
    #   3. Save the new checkpoint.
    #   4. If `save_top_k` is set to 1:
    #      a. Check if the previously saved checkpoint should be removed (to ensure only the latest is retained).
    #      b. If so, remove the previous checkpoint.

    def _save_none_monitor_checkpoint(self, trainer: 'flame_modules.Trainer', monitor_candidates: dict[str, torch.tensor]) -> None:
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer)
        #@: Set the best model path before saving because it will be a part of the state
        previous, self.best_model_path = self.best_model_path, filepath
        self._save_checkpoint(trainer, previous)
        
        if self.save_top_k == 1 and previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)
            
        
    
    
    
    # `_update_best_and_save` Method in ModelCheckpoint Class
    # --------------------------------------------------------
    # Purpose:
    #     This method updates the list of top `k` checkpoints based on the current metric, 
    #     and then saves the checkpoint if it qualifies as one of the top `k` checkpoints.
    #     It also manages the deletion of checkpoints when the limit of top `k` checkpoints 
    #     is reached, ensuring that only the best checkpoints (based on the specified metric) 
    #     are retained.
    #
    # Parameters:
    #   - `current`: The current value of the monitored metric.
    #   - `trainer`: An instance of `flame_modules.Trainer` that provides details 
    #                about the current training state.
    #   - `monitor_candidates`: A dictionary containing metrics which can be used in the file name.
    #
    # Key Steps:
    #   1. Determine the number of checkpoints to retain (`k`).
    #   2. If we already have `k` top checkpoints, identify the checkpoint to delete (`del_filepath`).
    #   3. Handle NaN metric values by replacing them with positive or negative infinity, based on the mode.
    #   4. Compute the filepath for the new checkpoint.
    #   5. Update the current score and the best `k` checkpoints dictionary.
    #   6. Update paths and values for the best and kth best checkpoints.
    #   7. Display a log message (if verbose) about the checkpoint being saved.
    #   8. Save the checkpoint.
    #   9. Remove any checkpoint identified for deletion in step 2 (if applicable).
    
    def _update_best_and_save(self, current: torch.tensor, trainer: 'flame_modules.Trainer', monitor_candidates: dict[str, torch.tensor]) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k
        del_filepath = None
        
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)
            
        #@: Do not save NaN, replace with +/- inf
        if isinstance(current, torch.tensor) and torch.isnan(current):
            current = torch.tensor(float('inf' if self.mode == 'min' else '-inf'), device= current.device)
        
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        
        #@: Save the current score
        self.current_score = current
        self.best_k_models[filepath] = current
        
        if len(self.best_k_models) == k:
            #@: monitor dict has reached `k` elements
            _op = max if self.mode == 'min' else min
            self.kth_best_model_path = _op(self.best_k_models, key= self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]
        
        _op = min if self.mode == 'min' else max
        self.best_model_path = _op(self.best_k_models, key= self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]
        
        if self.verbose:
            epoch = monitor_candidates['epoch']
            step = monitor_candidates['step']
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
            
        self._save_checkpoint(trainer, filepath)
        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            self._remove_checkpoint(trainer, del_filepath)
            
            
            
    
    
    # `to_yaml` Method in ModelCheckpoint Class
    # -----------------------------------------
    # Purpose:
    #     This method serializes the best `k` models' scores to a YAML file. The YAML 
    #     file will contain file paths of the top `k` checkpoints as keys and their 
    #     corresponding scores as values.
    #
    # Parameters:
    #   - `filepath`: Optional path to the location where the YAML file should be saved. 
    #                 If not provided, it uses the default directory set in the `ModelCheckpoint` class.
    #
    # Key Steps:
    #   1. Convert tensor values in `best_k_models` to native Python float values.
    #   2. Determine the file path for the YAML file if not provided.
    #   3. Write the `best_k` dictionary to the YAML file.

    def to_yaml(self, filepath: Optional[_PATH] = None) -> None:
        best_k = {k: v.item() for k, v in self.best_k_models.items()}
        if filepath is None:
            assert self.dirpath
            filepath = os.path.join(self.dirpath, 'best_k_models.yaml')
        
        with self._fs.open(filepath, 'w') as fp:
            yaml.dump(best_k, fp)





    def file_exists(self, filepath: _PATH, trainer: 'flame_modules.Trainer') -> bool:
        #@: Checks if a file exists on rank 0 and broadcasts the result to all other ranks, preventing the internal
        #@: state to diverge between ranks.
        exists = self._fs.exists(filepath)
        return trainer.strategy.broadcast(exists)





    # `_should_remove_checkpoint` Method in ModelCheckpoint Class
    # -----------------------------------------------------------
    # Purpose:
    #     Determines whether a previously saved checkpoint file should be deleted or not based on
    #     various conditions.
    #
    # Parameters:
    #   - `trainer`: An instance of the `Trainer` class from the `flame_modules` module.
    #   - `previous`: File path to the previously saved checkpoint.
    #   - `current`: File path to the current checkpoint.
    #
    # Key Steps:
    #   1. Return `False` if the previous checkpoint file path is the same as the current one.
    #   2. If the filesystem is not local (i.e., not 'file'), return `True` to remove the checkpoint.
    #   3. Determine the absolute path of the previous checkpoint and the trainer's resume checkpoint.
    #   4. Return `False` if the previous checkpoint is the one the trainer resumed from.
    #   5. Check if the previous checkpoint exists within the directory path of the current checkpoints.
    #      Return `True` if it exists in the directory, indicating it should be removed.

    def _should_remove_checkpoint(self, trainer: 'flame_modules.Trainer', previous: str, current: str) -> bool:
        #@: Checks if the previous checkpoint should be deleted.
        # A checkpoint won't be deleted if any of the cases apply:
        #   * The previous checkpoint is the same as the current checkpoint
        #   * The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        #   * The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        if previous == current:
            return False
        if self._fs.protocol != 'file':
            return True
        
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None
        if resume_path is not None and previous == resume_path:
            return False
        
        assert self.dirpath is not None
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents




    def _remove_checkpoint(self, trainer: 'flame_modules.Trainer', filepath: str) -> None:
        trainer.strategy.remove_checkpoint(filepath)     
        
            

    
    
    