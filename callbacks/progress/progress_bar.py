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
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


# ------------------------------
# ProgressBar in FlameModules
# ------------------------------
#
# What is ProgressBar?
#
#     ProgressBar is a class in the FlameModules library that serves as a visual indicator of training 
#     progress during model fitting. It inherits from the `Callback` class and is designed to track and 
#     display the progress of batch processing within the `Trainer` class.
#
# Why do we need ProgressBar?
#
#     1. Visual Feedback: Provides immediate visual feedback about the training process, which is crucial for 
#                         long-running training sessions.
#
#     2. User Engagement: Helps to keep the user informed about the training progress, preventing the sense 
#                         of a "black box" experience while waiting for the training to complete.
#
#     3. Estimation of Completion Time: Can help in estimating how much time is left until the training 
#                                      process is completed.
#
#     4. Detecting Stalls and Hangs: A static progress bar might indicate that the training process is 
#                                    stalled or hanging, which is useful for debugging.
#
#     5. Monitoring Training Phases: Can be used to monitor different phases of training, such as epoch 
#                                    completion and validation checks.
#
#
# How does ProgressBar work?
#
#     - When a `Trainer` instance starts a training loop, the `ProgressBar` callback is triggered to update 
#       itself at each step or batch processed.
#
#     - It typically displays a visual bar that fills up as the batch progresses, along with additional 
#       information such as the percentage completed, the current batch number, and other relevant metrics.
#
#     - The progress bar updates are often printed to the console or integrated into a web-based dashboard 
#       in more sophisticated environments.
#
#
# The Functionality of ProgressBar in FlameModules:
#
#     - As a subclass of `Callback`, `ProgressBar` can override callback methods to execute its updates 
#       at specific points in the training loop (e.g., on_batch_end, on_epoch_end).
#
#     - It can be customized or extended to provide additional functionality, such as custom messages or 
#       progress indicators tailored to the specific needs of the user or the training process.
#
#     - The ProgressBar is an optional feature that can be added or removed from the `Trainer` callbacks 
#       list according to the user's preference for having a progress indicator.
#
#
# The importance of the ProgressBar class within FlameModules lies in its role in enhancing the user experience during 
# model training, by providing a simple yet effective way to monitor the training progress. It adds a layer of transparency 
# to the training process, which can be essential for debugging, optimization, and user satisfaction.


class ProgressBar(Callback):
    #@: The base class for progress bars in `FLameModules`. It is a :class: `~FlameModules.callbacks.callback.Callback` that keeps
    # track of the batch progress in the :class: `~FlameModules.trainer.trainer.Trainer`. 
     
    def __init__(self) -> None:
        self._trainer: Optional['flame_modules.Trainer'] = None
        self._current_eval_dataloader_index: Optional[int] = None
        
        
    
    @property
    def trainer(self) -> 'flame_modules.Trainer':
        if self._trainer is None:
            raise TypeError(f'The `{self.__class__.__name__}._trainer` reference has not been set yet.')
        return self._trainer
    
    
    
    @property
    def sanity_check_description(self) -> str:
        return 'Sanity Checking'
    
    
    
    @property
    def train_description(self) -> str:
        return 'Training'
    
    
    
    @property
    def validation_description(self) -> str:
        return 'Validation'
    
    
    
    @property
    def test_description(self) -> str:
        return 'Testing'
    
    
    @property
    def predict_description(self) -> str:
        return 'Predicting'
    
    
    
    @property
    def total_train_batches(self) -> Union[int, float]:
        return self.trainer.num_training_batches
    
    
    
    @property
    def total_val_batches_current_dataloader(self) -> Union[int, float]:
        batches = self.trainer.num_sanity_val_batches if self.trainer.sanity_checking else self.trainer.num_val_batches
        if isinstance(batches, list):
            assert self._current_eval_dataloader_index is not None
            return batches[self._current_eval_dataloader_index]
        return batches
    
    
    @property
    def total_test_batches_current_dataloader(self) -> Union[int, float]:
        batches = self.trainer.num_test_batches
        if isinstance(batches, list):
            assert self._current_eval_dataloader_index is not None
            return batches[self._current_eval_dataloader_index]
        return batches
    
    
    
    @property
    def total_predict_batches_current_dataloader(self) -> Union[int, float]:
        assert self._current_eval_dataloader_index is not None
        return self.trainer.num_predict_batches[self._current_eval_dataloader_index]
    
    
    
    @property
    def total_val_batches(self) -> Union[int, float]:
        if not self.trainer.fit_loop.epoch_loop._should_check_val_epoch():
            return 0
        
        return (
            sum(self.trainer.num_val_batches) if isinstance(self.trainer.num_val_batches, list) else self.trainer.num_val_batches
        )
        
        
    
    
    def has_dataloader_changed(self, dataloader_index: int) -> bool:
        old_dataloader_index: int = self._current_eval_dataloader_index
        self._current_eval_dataloader_index = dataloader_index
        return old_dataloader_index != dataloader_index
    
    
    
    def reset_dataloader_index_tracker(self) -> None:
        self._current_eval_dataloader_index = None
        
        
        
    def disable(self) -> None:
        raise NotImplementedError
    
    
    
    
    def enable(self) -> None:
        raise NotImplementedError
    
    
    
    
    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)
        
        
        
    
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        self._trainer = trainer
        if not trainer.is_global_zero:
            self.disable()
            
            
    
    def get_metrics(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> dict[str, Union[int, str, float, dict[str, float]]]:
        standard_metrics = get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
             rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )
        
        return {**standard_metrics, **pbar_metrics}
    
    


    def get_standard_metrics(trainer: 'flame_modules.Trainer') -> dict[str, Union[int, str]]:
        items_dict: dict[str, Union[int, str]] = {}
        if trainer.loggers:
            from lightning.pytorch.loggers.utilities import _version
            
            if (version != _version(trainer.loggers)) not in ('', None):
                if isinstance(version, str):
                    version = version[-4:]
                items_dict['v_num'] = version
                
        return items_dict
    
    
    
            
    
    
    
    
    
#@: Driver Code
if __name__.__contains__('__main__'):
    print('hemllo')