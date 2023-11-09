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
import logging, os, torch
from argparse import Namespace
from typing_extensions import override

from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from lightning.fabric.loggers.tensorboard import TensorBoardLogger as FabricTensorBoardLogger
from lightning.fabric.utilities.cloud_io import _is_dir
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.types import _PATH

from FlameModules.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.core.saving import save_hparams_to_yaml
from FlameModules.loggers.logger import Logger
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn


log = logging.getLogger(__name__)


# ----------------------------------
# TensorBoardLogger in FlameModules
# ----------------------------------
# What is TensorBoardLogger?
#
#     The TensorBoardLogger is an integral component of FlameModules that interfaces with TensorBoard, 
#     a visualization toolkit for machine learning experimentation. It facilitates the logging of 
#     parameters, metrics, and potentially computational graph structures, allowing for detailed 
#     monitoring and analysis of the training process.
#
# Why do we need TensorBoardLogger in FlameModules?
#
#     1. Insightful Visualization: It provides a window into the model's training and validation metrics, 
#                                  making the analysis of complex data more intuitive and actionable.
#
#     2. Hyperparameter Tuning: By logging hyperparameters and their corresponding outcomes, it aids 
#                                in fine-tuning the model's performance for better results.
#
#     3. Model Understanding: The optional graph logging feature gives an in-depth look at the model architecture, 
#                             helping developers understand and improve their models.
#
#     4. Track and Compare: TensorBoardLogger enables tracking multiple runs simultaneously for comparison, 
#                           which is crucial for experimenting with different model architectures and configurations.
#
#     5. Easy-to-use Interface: With a straightforward initialization process that links with FlameModules, 
#                               the logger provides a hassle-free setup for advanced logging capabilities.
#
# How to use TensorBoardLogger in FlameModules?
#
#     Instantiate the TensorBoardLogger with a specified save directory, optional run name, version, 
#     and other configurations. Upon integration into FlameModules' Trainer, it automatically handles 
#     the logging of specified metrics and parameters to TensorBoard. Developers can view and analyze 
#     these logs through the TensorBoard interface, gaining valuable insights into the model's performance 
#     and behavior throughout the training lifecycle.
#

class TensorBoardLogger(Logger, FabricTensorBoardLogger):
    name_hparams_file: str = 'hparams.yaml'
    
    def __init__(self, save_dir: _PATH, name: Optional[str] = 'flame_logs', version: Optional[Union[int, str]] = None, log_graph: Optional[bool] = False, 
                                                                                                                       default_hp_metric: Optional[bool] = True, 
                                                                                                                       prefix: Optional[str] = '', 
                                                                                                                       sub_dir: Optional[_PATH] = None, 
                                                                                                                       **kwargs: Any) -> None:
        # Purpose:
        #     Constructs the TensorBoardLogger for integration with FlameModules training processes. This logger
        #     sets up the logging directory and configuration, prepares to record hyperparameters, and manages 
        #     conditions for graph logging based on TensorBoard availability.
        #
        # Parameters:
        #   - `save_dir`: The base directory where logs and outputs will be saved.
        #   - `name`: An optional name for the log directory. If not provided, defaults to 'flame_logs'.
        #   - `version`: An optional version number or string. This can be used to differentiate between different runs.
        #   - `log_graph`: An optional flag to indicate if the computational graph should be logged. Defaults to False.
        #   - `default_hp_metric`: A boolean flag to include a default hyperparameter metric. Defaults to True.
        #   - `prefix`: An optional prefix for all keys logged. Useful when grouping logs.
        #   - `sub_dir`: An optional subdirectory within the main `save_dir`.
        #
        # Key Steps:
        #   1. Call the superclass constructor to set up the logger with provided directory and configuration details.
        #   2. Validate the `log_graph` flag against the availability of TensorBoard. Warn if TensorBoard is not available but graph logging is requested.
        #   3. Assign the effective `log_graph` flag to the `_log_graph` attribute, determining if graph logging should proceed.
        #   4. Initialize an empty dictionary or Namespace for hyperparameters (`hparams`), to be populated later during training.
        #
        # Example Usage:
        #   import FlameModules as flame_modules
        #   from flame_modules.trainer.trainer import Trainer
        #   from flame_modules.core.flame_module import FlameModule
        #   from flame_modules.loggers.tensorboard import TensorBoardLogger
        #
        #   class FlameModel(FlameModule):
        #       ...
        #
        #   if __name__.__contains__('__main__'):
        #       model = FlameModel()
        #       tb_logger = TensorBoardLogger(save_dir= './logs', name= 'temp_project', log_graph= True)
        #       trainer = Trainer(logger= tb_logger)
        #       trainer.fit(model)
        #
        # Importance:
        #   This constructor method is crucial for enabling detailed logging and visualization capabilities within the FlameModules framework. By providing 
        #   a structured way to initiate logging, it streamlines the process of tracking and analyzing model performance, leading to more informed 
        #   decision-making and potentially better-performing models.
        
        super(TensorBoardLogger, self).__init__(
            root_dir= save_dir,
            name= name,
            version= version,
            default_hp_metric= default_hp_metric,
            prefix= prefix,
            sub_dir= sub_dir,
            **kwargs
        )
        if log_graph and not _TENSORBOARD_AVAILABLE:
            rank_zero_warn(
                'You set `TensorBoardLogger(log_graph= True)` but `tensorboard` is not available.\n'
                f'{str(_TENSORBOARD_AVAILABLE)}'
            )
            
        self._log_graph = log_graph and _TENSORBOARD_AVAILABLE
        self.hparams: Union[dict[str, Any], Namespace] = {}
        
        
        
    @property
    @override
    def root_dir(self) -> str:
        return os.path.join(super().root_dir, self.name)
    
    
    @property
    @override
    def log_dir(self) -> str:
        version: str = self.version if isinstance(self.version, str) else f'version_{self.version}'
        log_dir: str = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir: str = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir
    
    
    
    @property
    @override
    def save_dir(self) -> str:
        return self._root_dir
    
    
    
    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], metrics: Optional[dict[str, Any]] = None) -> None:
        if _OMEGACONF_AVAILABLE:
            from omegaconf import Container, OmegaConf
            
        params = _convert_params(params)
        
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)
            
        return super().log_hyperparams(params= params, metrics= metrics)
    
    
    
    @override
    @rank_zero_only
    def log_graph(self, model: 'flame_modules.FlameModule', input_array: Optional[torch.tensor] = None) -> None:
        if not self._log_graph:
            return 
        
        input_array: torch.tensor = model.example_input_array if input_array is None else input_array
        
        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
            
        elif not isinstance(input_array, (torch.tensor, tuple)):
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `input_array` or `model.example_input_array`"
                f" has type {type(input_array)} which can't be traced by TensorBoard. Make the input array a tuple"
                f" representing the positional arguments to the model's `forward()` implementation."
            )
            
        else:
            input_array: torch.tensor = model._on_before_batch_transfer(input_array)
            input_array: torch.tensor = model._apply_batch_transfer_handler(input_array)
            with lightning.pytorch.core.module._jit_is_scripting():
                self.experiment.add_graph(model, input_array)
                
                
    
    @override
    @rank_zero_only
    def save(self) -> None:
        super().save()
        dir_path: str = self.log_dir
        hparams_file = os.path.join(dir_path, self.name_hparams_file)
        
        if _is_dir(self._fn, dir_path) and not self._fn.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)
            
            
    
    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if status == 'success':
            self.save()
            
            
    
    
    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        #@: Called after model checkpoint callback saves a new checkpoint.
        ...
        
        
    
    @override
    def _get_next_version(self) -> int:
        root_dir: str = self.root_dir
        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            log.warning(f'Missing logger folder: {root_dir}')
            return 0
        
        existing_versions: list[Any] = []
        for listing in listdir_info:
            d = listing['name']
            bn = os.path.basename(d)
            if _is_dir(self._fs, d) and bn.startswith('version_'):
                dir_ver = bn.split('_')[1].replace('/', '')
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        
        if len(existing_versions) == 0:
            return 0
        
        return max(existing_versions) + 1
    
  