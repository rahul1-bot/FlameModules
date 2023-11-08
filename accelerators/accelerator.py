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
from abc import ABC
from lightning.fabric.accelerators.accelerator import Accelerator as _Accelerator
import torch

# -------------------------------
# Accelerator in FlameModules
# -------------------------------
#
# What is an Accelerator?
#
#     An Accelerator in the FlameModules framework is an abstraction that simplifies the 
#     use of different hardware for training models. It ensures that the model can be run 
#     on CPUs, GPUs, TPUs, or other hardware without changing the underlying model code.
#
#
# Why do we need an Accelerator?
#
#     1. Hardware Abstraction: To manage the complexity of running models on different 
#        hardware with a unified interface.
#
#     2. Efficiency and Scalability: To ensure efficient utilization of hardware for 
#        better performance and scalability of model training.
#
#     3. Ease of Use: To allow developers to run models on various hardware without 
#        extensive knowledge about the specifics of each environment.
#
#     4. Flexibility: To provide the ability to choose the best hardware for specific 
#        tasks without being locked into one type of hardware.
#
#     5. Performance Optimization: To benefit from hardware-specific optimizations 
#        that improve model performance without extra development work.
#
#
# How does an Accelerator work?
#
#     - The Accelerator configures the training environment to match the model's requirements 
#       and the available hardware resources.
#     - It manages the data flow between the CPU and other hardware to maximize throughput 
#       and efficiency.
#     - The Accelerator is also responsible for gathering hardware statistics and profiling, 
#       which are essential for optimization and resource management.
#
#
# Role of Accelerator in FlameModules:
#
#     The Accelerator class in FlameModules is a base class from which specific accelerator 
#     implementations are derived. It provides essential methods for environment setup and 
#     device statistics.
#     - The `setup` method prepares the trainer and the training environment.
#     - The `get_device_stats` method is intended to be implemented by subclasses to provide 
#       hardware-specific statistics for debugging and optimization.
#
#
# In essence, the Accelerator class serves as the backbone for hardware management in the 
# FlameModules framework, facilitating the efficient and flexible use of computational resources 
# for model training across different environments.

class Accelerator(_Accelerator, ABC):
    def setup(self, trainer: 'flame_modules.Trainer') -> None:
        ...
        
        
    
    def get_device_stats(self, device: torch.device) -> dict[str, Any]:
        raise NotImplementedError
    
    
