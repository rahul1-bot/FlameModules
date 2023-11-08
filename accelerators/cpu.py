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
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning.fabric.utilities.types import _DEVICE
from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.cpu import _parse_cpu_cores
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


# --------------------------------
# CPUAccelerator in FlameModules
# --------------------------------

# Purpose:
#     The `CPUAccelerator` class is a concrete implementation of the `Accelerator` base class, 
#     specialized for CPU operations in the FlameModules framework. It ensures that the model
#     training and other computations are correctly configured to run on CPU hardware.

# Methods:
#   - `setup_device`: 
#       Configures the specific device for the training process. If a non-CPU device is provided, 
#       it raises a `MisconfigurationException` to enforce the use of CPU.

#   - `get_device_stats`: 
#       Returns a dictionary containing statistics and information about the CPU, such as utilization 
#       and memory usage, by calling `get_cpu_stats()` function.

#   - `teardown`: 
#       Handles any cleanup operations that need to be performed when the training process on the CPU 
#       is complete.

#   - `parse_devices`: 
#       A static method that interprets the provided device descriptors and ensures a list of CPU 
#       devices is returned, allowing for potential multi-core CPU training setups.

#   - `register_accelerators`: 
#       A class method to register the `CPUAccelerator` within the provided `accelerator_registry`, 
#       making it available for use within the FlameModules ecosystem.

# Parameters:
#   - `device`: An instance of `torch.device` that specifies the type of device to configure.
#   - `devices`: Can be an integer, string, or a list of integers specifying the CPU cores.
#   - `accelerator_registry`: A registry to which the `CPUAccelerator` adds itself for availability.

# Returns:
#   - For `get_device_stats`: Returns CPU statistics.
#   - For `parse_devices`: Returns a list of configured CPU devices.

# Key Steps:
#   1. The `setup_device` method ensures the model is set up to run on CPU.
#   2. The `get_device_stats` method gathers current CPU statistics for monitoring and optimization.
#   3. The `teardown` method is a placeholder for cleanup procedures after training.
#   4. The `parse_devices` method decodes the desired CPU configuration into a valid list of devices.
#   5. The `register_accelerators` method makes the `CPUAccelerator` known to the `accelerator_registry`.

# Importance:
#   - This class is crucial for those who want to train models on CPUs, ensuring compatibility and optimal use 
#     of available CPU resources.
#   - It provides clear interfaces for CPU device management and contributes to the overall modularity and 
#     flexibility of the FlameModules framework.

class CPUAccelerator(Accelerator):
    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != 'cpu':
            raise MisconfigurationException(f'Device should be CPU, got {device} instead.')
        
        
    
    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        return get_cpu_stats()
    
    
    
    @override
    def teardown(self) -> None:
        ...
        
        
        
    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> list[torch.device]:
        devices = _parse_cpu_cores(devices)
        return [torch.device('cpu')] * devices
    
    
    
    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register('cpu', cls, description= cls.__name__)



#@: CPU device metrics
_cpu_vm_percent: str = 'cpu_vm_percent'
_cpu_percent: str = 'cpu_percent'
_cpu_swap_percent: str = 'cpu_swap_percent'
_psutil_available = RequirementCache('psutil')

# Collects and returns CPU statistics, including virtual memory usage, CPU usage, and swap memory usage.
# Requires 'psutil' package to retrieve system-level information. If 'psutil' isn't available, it raises `ModuleNotFoundError`.

def get_cpu_stats() -> dict[str, float]:
    if not _psutil_available:
        raise ModuleNotFoundError(
            f"Fetching CPU device stats requires `psutil` to be installed. {str(_psutil_available)}"
        )
    import psutil
    return {
        _cpu_vm_percent: psutil.virtual_memory().percent,
        _cpu_percent: psutil.cpu_percent(),
        _cpu_swap_percent: psutil.swap_memory().percent
    }
    
