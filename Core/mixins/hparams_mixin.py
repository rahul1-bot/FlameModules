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
import copy, inspect, types
from argparse import Namespace


__note: str = r'''
    The HyperparametersMixin class offers several benefits when integrated into a machine learning framework or pipeline, 
    especially in the context of a PyTorch-based framework. Here are some of the advantages:

    * Centralized Hyperparameter Management: 
            Provides a unified place to manage, retrieve, and set hyperparameters, helping ensure consistency across the framework.

    * Immutable Initial Hyperparameters:
            The hparams_initial property allows users to reference the original set of hyperparameters, regardless of any subsequent 
            modifications. This is crucial for reproducibility and understanding any changes made during the model's training or tuning.

    * Flexibility in Input Types:
            The class can handle hyperparameters provided as dictionaries, namespaces, or other allowed types, offering flexibility 
            to the user in terms of input format.

    * Type Safety:
            By restricting hyperparameters to specific types (e.g., disallowing primitive types), the class ensures that the hyperparameters 
            are structured and organized.

    * Ease of Integration:
            Being a mixin, this class can be easily integrated into other classes without the need for inheritance, thereby providing 
            hyperparameter functionality to any class it's mixed into.

    * Encapsulation:
            The class encapsulates the logic for hyperparameter parsing and storage, abstracting away the complexity and providing a 
            clean interface to the end-user.

    * Error Handling:
            The class includes checks and raises errors for unsupported hyperparameter types, guiding users to provide inputs in the 
            correct format.

    * Integration with Logging:
            The _log_hyperparams flag and related logic suggest that the class can be easily integrated with logging mechanisms to 
            track changes and updates to hyperparameters, which is crucial for experiment tracking.

    * Just-In-Time (JIT) Compilation Support:
            The __jit_unused_properties__ list indicates consideration for JIT compilation, ensuring the class plays nicely with 
            PyTorch's JIT features.


    In essence, the HyperparametersMixin class provides a structured and efficient way to handle hyperparameters in a machine learning 
    workflow, promoting best practices, consistency, and reproducibility.
'''

__summary: str = r'''
    The HyperparametersMixin provides a structured way to handle hyperparameters. It ensures:
        * Hyperparameters are stored in a unified format, no matter their original format.
        * Hyperparameters can be accessed easily.
        * The initial values of hyperparameters remain immutable and accessible.

This kind of structure is beneficial when there's a need to track, modify, or log hyperparameters during model training or experimentation. 
The mixin abstracts away the complexities and provides a clean, user-friendly interface for handling hyperparameters.

'''
# import torch.nn as nn

# class SimpleNN(HyperparametersMixin, nn.Module):  # Notice the mixin is used alongside nn.Module
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNN, self).__init__()  # Call the super constructor
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
        
#         # Save the hyperparameters
#         self.save_hyperparameters('input_size', 'hidden_size', 'output_size')
        
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# model = SimpleNN(784, 128, 10)


# # Access hyperparameters
# print(model.hparams)  # AttributeDict with input_size, hidden_size, and output_size

# # If you ever modify the hyperparameters, the initial values are still retained:
# model.hparams['hidden_size'] = 256
# print(model.hparams_initial)  # This will still show hidden_size as 128

# The HyperparametersMixin allows the SimpleNN model to easily manage and access its hyperparameters in a structured manner, without a
# dding much complexity to the model's core functionality. This is just a basic example, but in a larger, more complex project, the 
# benefits of using such a mixin for hyperparameter management become even more apparent.


import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
import lightning
from lightning.pytorch.utilities.parsing import AttributeDict, save_hyperparameters



class HyperparamertersMixin:
    __jit_unused_properties__: list[str] = [
        'hparams', 'hparams_initial'
    ]
    
    def __init__(self) -> None:
        super().__init__()
        self._log_hyperparams: bool = False
        
        
    def save_hyperparameters(self, *args: Any, ignore: Optional[Union[Sequence[str], str]] = None,
                                               frame: Optional[types.FrameType] = None,
                                               logger: Optional[bool] = True) -> None:
        
        self._log_hyperparams = logger
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        
        save_hyperparameters(self, *args, ignore= ignore, frame= frame)
        
    
    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        hp = self._to_hparams_dict(hp)
        
        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp
            
    
    @staticmethod
    def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, _PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {_PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, _ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")
        return hp


    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
            
        return self._hparams



    @property
    def hparams_initial(self) -> AttributeDict:
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        
        # prevent any change
        return copy.deepcopy(self._hparams_initial)



