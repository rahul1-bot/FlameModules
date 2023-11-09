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
import functools, operator
from abc import ABC
from collections import defaultdict
from typing import Mapping, Sequence
import numpy as np
from typing_extensions import override
from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment 
from lightning.fabric.loggers.logger import rank_zero_experiment 
from FlameModules.callbacks.model_checkpoint import ModelCheckpoint

# -----------------------
# Logger in FlameModules
# -----------------------
#
# What is a Logger?
#
#     A Logger in the FlameModules framework is a component that records events, metrics, and 
#     artifacts during the model training process. It is crucial for tracking the progress of 
#     training, debugging, and analyzing the performance of models.
#
# Why do we need a Logger?
#
#     1. Experiment Tracking: To keep a record of model training sessions, including parameters,
#        metrics, and outputs.
#     2. Debugging: To provide insights into the training process and help identify any issues.
#     3. Reproducibility: To ensure that experiments can be reproduced and results can be verified.
#     4. Performance Analysis: To analyze and compare different training runs for optimization.
#     5. Collaboration: To share training progress and results with others.
#
# How does a Logger work?
#
#     - The Logger captures and records information about the training process, such as hyperparameters,
#       training loss, validation metrics, model checkpoints, and other relevant data.
#     - It may save this information in various formats like logs, CSV files, or databases, and 
#       may provide interfaces for visualization and analysis.
#     - Some Logger implementations can integrate with web services to provide remote tracking and
#       notifications.
#
# Role of Logger in FlameModules:
#
#     The Logger class in FlameModules is an abstract base class that defines the interface for 
#       experiment loggers within the ecosystem.
#     - The `after_save_checkpoint` method provides a hook that is called after a model checkpoint
#       is saved, which can be used to log the event or perform related actions.
#     - The `save_dir` property can be overridden to specify the directory where logs should be saved.
#       If not overridden, it defaults to `None`, indicating that the logger does not save data locally.
#
# In summary, the Logger class is the foundation for logging in the FlameModules framework, enabling 
# detailed monitoring and analysis of the model training process. It provides a standard interface for 
# different types of logging implementations, allowing flexibility and customization according to the 
# needs of the experiment.


class Logger(FabricLogger, ABC):
    #@: Base class for experiment loggers. 
    
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        ...
        
        
    @property
    def save_dir(self) -> Optional[str]:
        #@: Returns the root directory where experiment logs get saved, or `None` if the logger does 
        # not save data locally.
        return None
    
    

class DummyLogger(Logger):
    # Purpose:
    #     The `DummyLogger` class acts as a stand-in for other Logger types within the FlameModules framework.
    #     It provides a no-operation (no-op) logger that can be used when logging is to be bypassed or disabled, 
    #     ensuring that user code executes without errors even when logging is not required or desired.
    #
    # Parameters:
    #   - None required for initialization.
    #
    # Key Components:
    #   - `experiment`: A `DummyExperiment` property that mimics an experiment without performing any actions.
    #   - `log_metrics`: A no-op method to comply with the Logger interface without recording metrics.
    #   - `log_hyperparams`: Similarly a no-op method to comply with the Logger interface without recording hyperparameters.
    #   - `name`: An overridden property that returns an empty string, as the DummyLogger does not have a specific name.
    #   - `version`: An overridden property that returns an empty string, indicating no versioning is applicable.
    #
    # Behavior:
    #   - The `DummyLogger` overrides the necessary methods and properties to ensure it can be substituted in place of a 
    #     functional logger without causing attribute errors.
    #   - It acts as a placeholder, allowing user code that expects a logger to function correctly even when logging is 
    #     turned off or not needed.
    #   - The overridden `__getitem__` and `__getattr__` methods ensure that any attempted attribute access or method call 
    #     on the `DummyLogger` does not result in an AttributeError, further adding to its utility as a no-op stand-in.
    #
    # Usage:
    #   - The `DummyLogger` is particularly useful during testing, when logging is not the focus, or when certain features 
    #     need to run without the overhead of actual logging.
    #   - It can also be used to disable user's loggers temporarily without modifying the user's original codebase.
    #
    def __init__(self) -> None:
        super(DummyLogger, self).__init__()
        self._experiment = DummyExperiment()
        
        
    @property
    def experiment(self) -> DummyExperiment:
        return self._experiment
    
    
    
    @override
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        ...
        
        
        
    @override
    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        ...
        
        
    
    @property
    @override
    def name(self) -> str:
        return ''
    
    
    @property
    @override
    def version(self) -> str:
        return ''
    
    
    
    def __getitem__(self, index: int) -> 'DummyLogger':
        return self
    
    
    
    def __getattr__(self, name: str) -> Callable[Any]:
        #@: Allows the `DummyLogger` to be called with arbitrary methods, to avoid AttributeErrors
        def method(*args: Any, **kwargs: Any) -> None:
            return None
        
        return method
    
