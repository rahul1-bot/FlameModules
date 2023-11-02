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
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.utilities.exceptions import MisconfigurationException

# ----------------------------------
# DeviceStatsMonitor in FlameModules
# ----------------------------------
# What is DeviceStatsMonitor?
#
#     DeviceStatsMonitor is a callback utility specifically developed for FlameModules, a unique deep learning framework. 
#     Its fundamental purpose is to routinely monitor and log essential device statistics during the stages of training, 
#     validation, and testing. Notably, the DeviceStatsMonitor necessitates a logger to be associated as an argument 
#     with the Trainer. This ensures that all device metrics are systematically logged, thereby offering a transparent 
#     view of the device's operational behavior.
#
# Why do we need DeviceStatsMonitor in FlameModules?
#
#     1. Comprehensive Insights: It proffers intricate details into how the device is operating, shedding light on 
#                                metrics like GPU consumption, memory distribution, and potential areas that could 
#                                be causing slowdowns in the training.
#
#     2. Simplify Debugging: Having real-time access to device stats equips developers with the capability to 
#                             promptly identify and resolve performance issues, leading to optimized model training.
#
#     3. Eliminate Manual Logging: Manual recording of device statistics can be tedious. DeviceStatsMonitor, by 
#                                  automating this process, ensures developers can consistently oversee device 
#                                  performance without the hassle of manual logging.
#
#     4. Optimal Model Performance: By leveraging insights from device metrics, developers can make strategic 
#                                   decisions that not only refine model performance but also expedite training durations.
#
#     5. Seamless Callback Integration: As a callback, integrating it within any training workflow in the FlameModules 
#                                      ecosystem is straightforward, making it a versatile tool for a diverse range of 
#                                      projects.
#
# How to use DeviceStatsMonitor in FlameModules?
#
#     Initialization of the DeviceStatsMonitor requires a logger to be passed to the FlameModule's Trainer. Once in place, 
#     the callback operates autonomously, diligently monitoring and logging device stats throughout the training, 
#     validation, and testing phases. Subsequently, the aggregated metrics can be accessed and scrutinized via the 
#     designated logger.



class DeviceStatsMonitor(Callback):
    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        # Purpose:
        #     Initializes the DeviceStatsMonitor callback to monitor device statistics during training.
        #
        # Parameters:
        #   - `cpu_stats`: A boolean flag indicating if CPU statistics should be monitored. By default, it's set to None.
        #
        # Key Steps:
        #   1. Assign the `cpu_stats` parameter to the `_cpu_stats` attribute for the instance.
        
        self._cpu_stats = cpu_stats
        
    
    
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        # Purpose:
        #     Sets up the DeviceStatsMonitor callback for monitoring device statistics, specifically during the fitting phase.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `stage`: The stage in the training process. If it's not 'fit', the method returns early.
        #
        # Key Steps:
        #   1. Check if the current stage is the 'fit' phase.
        #   2. If the trainer does not have loggers, raise a ModuleNotFoundError, indicating that `psutil` is not installed, 
        #      which is required for logging CPU stats.

        if stage != 'fit':
            return
        
        if not trainer.loggers:
            raise ModuleNotFoundError(
                f"`DeviceStatsMonitor` cannot log CPU stats as `psutil` is not installed. {str(_PSUTIL_AVAILABLE)} "
            )
 
 
    
    def _get_and_log_device_state(self, trainer: 'flame_modules.Trainer', key: str) -> None:
        # Purpose:
        #     Retrieve device statistics (either CPU or GPU) and log them during training.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `key`: A key indicating the context in which the device statistics should be logged.
        #
        # Key Steps:
        #   1. Check if the logger connector should update logs. If not, return early.
        #   2. Get the root device (either CPU or GPU) from the trainer's strategy.
        #   3. If CPU stats logging is disabled and the device is a CPU, return early.
        #   4. Fetch the current device stats based on the selected device.
        #   5. If CPU stats are enabled and the device is not a CPU, gather CPU stats and add them to the device stats.
        #   6. Iterate over all trainer loggers and log the fetched device statistics with a prefixed metric key to 
        #      differentiate the context.

        if not trainer._logger_connector.should_update_logs:
            return 
        
        device: torch.device = trainer.strategy.root_device
        if self._cpu_stats is False and device.type == 'cpu':
            return 
        
        device_stats = trainer.accelerator.get_device_stats(device)
        
        if self._cpu_stats and device.type != 'cpu':
            from lightning.pytorch.accelerator.cpu import get_cpu_stats
            device_stats.update(get_cpu_stats())
            
        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(device_stats, f'{self.__class__.__qualname__}.{key}', separator)
            logger.log_metrics(prefixed_device_stats, step= trainer.fit_loop.epoch_loop._batches_that_stepped)
            
        
            
            
    def on_train_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, 
                                                                                                                batch_index: int) -> None:
        # Purpose:
        #     Hook that is called at the start of every training batch. Its primary purpose is to fetch and log device 
        #     statistics at the commencement of each training batch.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `batch`: The current batch data being processed in training.
        #   - `batch_index`: The index of the current batch.
        #
        # Key Steps:
        #   1. Call the `_get_and_log_device_state` method with the trainer instance and the context 'on_train_batch_start' 
        #      to fetch and log the device stats specifically at the start of the training batch.

        self._get_and_log_device_state(trainer, 'on_train_batch_start')
        
    
    
    
    
    def on_train_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', 
                                                                                                              batch: Any, 
                                                                                                              batch_index: int) -> None:
        # Purpose:
        #     Hook that is called at the end of every training batch. Its primary aim is to fetch and log device 
        #     statistics after the completion of each training batch.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `outputs`: The outputs from the training step (typically the loss).
        #   - `batch`: The current batch data that was processed in training.
        #   - `batch_index`: The index of the current batch.
        #
        # Key Steps:
        #   1. Call the `_get_and_log_device_state` method with the trainer instance and the context 'on_train_batch_end' 
        #      to fetch and log the device stats specifically at the end of the training batch.

        self._get_and_log_device_state(trainer, 'on_train_batch_end')
        
        
        
        
        
    def on_validation_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, 
                                                                                                                     batch_index: int, 
                                                                                                                     dataloader_index: Optional[int] = 0) -> None:
        # Purpose:
        #     Hook that is invoked at the commencement of every validation batch. Its primary objective is to 
        #     retrieve and log the device statistics as the validation batch begins.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `batch`: The current batch data that will be processed in validation.
        #   - `batch_index`: The index of the current validation batch.
        #   - `dataloader_index`: The index of the current dataloader (useful when multiple validation dataloaders are present). 
        #                          Default value is 0.
        #
        # Key Steps:
        #   1. Invoke the `_get_and_log_device_state` method with the trainer instance and the context 'on_validation_batch_start'
        #      to fetch and log the device stats specifically at the start of the validation batch.

        self._get_and_log_device_state(trainer, 'on_validation_batch_start')
        
        
        
        
        
    def on_validation_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', 
                                                                                                                   batch: Any, 
                                                                                                                   batch_index: int, 
                                                                                                                   dataloader_index: Optional[int] = 0) -> None:
        # Purpose:
        #     Hook that is invoked at the termination of every validation batch. Its primary aim is to 
        #     retrieve and log the device statistics as the validation batch concludes.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `outputs`: The result obtained after processing the current validation batch.
        #   - `batch`: The current batch data that has been processed in validation.
        #   - `batch_index`: The index of the current validation batch.
        #   - `dataloader_index`: The index of the current dataloader (relevant when multiple validation dataloaders are in play). 
        #                          Default value is 0.
        #
        # Key Steps:
        #   1. Invoke the `_get_and_log_device_state` method with the trainer instance and the context 'on_validation_batch_end'
        #      to fetch and log the device stats precisely at the close of the validation batch.

        self._get_and_log_device_state(trainer, 'on_validation_batch_end')
        
        
        
        
    
    def on_test_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, 
                                                                                                                           dataloader_index: Optional[int] = 0) -> None:
        # Purpose:
        #     Hook that is triggered at the onset of every testing batch. Its primary function is to 
        #     extract and log the device statistics as the testing batch commences.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `batch`: The current batch data that is set to be processed in testing.
        #   - `batch_index`: The index of the current testing batch.
        #   - `dataloader_index`: The index of the current dataloader (relevant when multiple testing dataloaders are in use). 
        #                          Default value is 0.
        #
        # Key Steps:
        #   1. Call the `_get_and_log_device_state` method with the trainer instance and the context 'on_test_batch_start'
        #      to fetch and record the device stats precisely as the testing batch begins.

        self._get_and_log_device_state(trainer, 'on_test_batch_start')
        
        
        
        
        
    def on_test_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', 
                                                                                                             batch: Any, 
                                                                                                             batch_index: int, 
                                                                                                             dataloader_index: Optional[int] = 0) -> None:
        # Purpose:
        #     Hook that is invoked at the conclusion of every testing batch. Its main role is to 
        #     retrieve and document the device statistics as the testing batch wraps up.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `outputs`: The result of the processing for the current testing batch, represented as 'Step_Output'.
        #   - `batch`: The current batch data that has been processed in testing.
        #   - `batch_index`: The index of the current testing batch.
        #   - `dataloader_index`: The index of the current dataloader (relevant when multiple testing dataloaders are in use). 
        #                          Default value is 0.
        #
        # Key Steps:
        #   1. Invoke the `_get_and_log_device_state` method with the trainer instance and the context 'on_test_batch_end'
        #      to extract and record the device stats precisely as the testing batch concludes.

        self._get_and_log_device_state(trainer, 'on_test_batch_end')
        
        
        
        
        
        
    def _prefix_metric_keys(metrics_dict: dict[str, float], prefix: str, separator: str) -> dict[str, float]:
        # Purpose:
        #     A utility function designed to prepend a specified prefix to the keys within a metrics dictionary.
        #
        # Parameters:
        #   - `metrics_dict`: A dictionary containing metrics, where each key represents a metric's name and 
        #                     its corresponding value represents the metric's value.
        #   - `prefix`: A string that will be prepended to each key within the metrics dictionary.
        #   - `separator`: A string that is used to separate the prefix from the original metric key. Common separators include 
        #                  underscores (_) or dots (.).
        #
        # Returns:
        #   - A new dictionary where each key is a combination of the prefix, separator, and the original metric key, 
        #     and each value is preserved from the original metrics dictionary.
        #
        # Key Steps:
        #   1. Iterate over each key-value pair in the `metrics_dict`.
        #   2. For each key, concatenate the `prefix`, `separator`, and the key itself.
        #   3. Return a new dictionary with the modified keys and the original values.

        return {
            prefix + separator + key : value
            for key, value in metrics_dict.items()
        }

