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

# Generate a Summary of all layers in a :class: `~FlameModules.core.FlameModule`.

# The String representation of this summary prints a table with columns containing
# the name, type and number of parameters for each layer.

import logging
from FlameModules.callbacks.callback import Callback
from lightning.pytorch.utilities.model_summary import DeepSpeedSummary, summarize
from lightning.pytorch.utilities.model_summary import ModelSummary as Summary
from lightning.pytorch.utilities.model_summary.model_summary import _format_summary_table

log = logging.getLogger(__name__)


# -------------------------------
# ModelSummary in FlameModules
# -------------------------------
#
# What is ModelSummary?
#
#     ModelSummary is a utility within the FlameModules framework that provides a high-level overview of a 
#     `FlameModule` (which represents a neural network model in FlameModules). The summary includes details 
#     about the layers of the network, such as the name, type, and number of parameters in each layer.
#
# Why do we need ModelSummary?
#
#     1. Understand Model Architecture: ModelSummary allows us to quickly understand the architecture of the 
#                                       neural network, including how layers are structured and connected.
#
#     2. Parameter Count Insights: Knowing the number of parameters at each layer can help diagnose potential 
#                                  issues with model complexity. A high number of parameters might imply a risk 
#                                  of overfitting, while too few can lead to underfitting.
#
#     3. Model Debugging: By summarizing the model, we can verify that the layers are correctly implemented and 
#                          troubleshoot issues related to model structure, which can be critical in the model 
#                          development phase.
#
#     4. Documentation and Reproducibility: ModelSummary can be used as a form of documentation to 
#                                           communicate the model’s architecture within a team or to ensure 
#                                           reproducibility in research settings.
#
#     5. Resource Management: Understanding the model's size and complexity can aid in managing computational 
#                             resources, as larger models generally require more memory and compute power.
#
#     6. Optimization and Refinement: By reviewing the model summary, developers can identify layers that may 
#                                     need optimization or refinement to improve model performance or efficiency.
#
# How does ModelSummary work?
#
#     - When invoked, ModelSummary scans through the `FlameModule` and records information about each layer.
#     - It organizes this information into a formatted table that is easy to read and interpret.
#     - The summary can be printed out or logged, providing a quick snapshot of the model at any stage of 
#       the development process.
#
# The String Representation of ModelSummary:
#
#     The string representation of the ModelSummary object, when printed, displays a neatly formatted table. 
#     Each row corresponds to a layer within the `FlameModule`, and the columns provide details such as the 
#     layer's name, its type (e.g., Conv2d, Linear, etc.), and the total number of trainable and non-trainable 
#     parameters. This tabular format makes it convenient for developers to review and understand the 
#     structural composition of the model.


class ModelSummary(Callback):
    #@: Code Doc:
    #       import FlameModules as flame_modules
    #       from flame_modules.trainer.trainer import Trainer
    #       from flame_modules.callbacks.model_summary import ModelSummary
    #
    #       if __name__.__contains__('__main__'):
    #           model_summary = ModelSummary(max_depth= 1)
    #           trainer = Trainer(callbacks= [model_summary])
    #
    # 
    def __init__(self, max_depth: Optional[int] = 1, **summarize_kwargs: Any) -> None:
        self._max_depth: int = max_depth
        self._summarize_kwargs: dict[str, Any] = summarize_kwargs
        
    
        
    def on_fit_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        if not self._max_depth:
            return 
        
        model_summary = self._summary(trainer, flame_module)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size
        
        if trainer.is_global_zero:
            self.summarize(summary_data, total_parameters, trainable_parameters, model_size, **self._summarize_kwargs)
            
    
    
    
    def _summary(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> Union[DeepSpeedSummary, Summary]:
        from lightning.pytorch.utilities.deepspeed import DeepSpeedStrategy
        
        if isinstance(trainer.strategy, DeepSpeedStrategy) and trainer.strategy.zero_stage_3:
            return DeepSpeedSummary(flame_module, max_depth= self._max_depth)
        
        return summarize(flame_module, max_depth= self._max_depth)
    
    
    
    @staticmethod
    def summarize(summary_data: list[tuple[str, list[str]]], total_parameters: int, trainable_parameters: int, model_size: float, **summarize_kwargs: Any) -> None:
        summary_table = _format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size, 
            *summary_data
        )
        log.info('\n' + summary_table)
        
        
