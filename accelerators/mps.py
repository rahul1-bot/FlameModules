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
from typing_extensions import override
from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.mps import MPSAccelerator as _MPSAccelerator
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from FlameModules.accelerators.accelerator import Accelerator
from FlameModules.accelerators.cpu import _psutil_available
from lightning.pytorch.utilities.exceptions import MisconfigurationException

# -------------------------------
# MPSAccelerator in FlameModules
# -------------------------------
#
# Purpose:
#     The `MPSAccelerator` class extends the `Accelerator` abstract base class, tailored for operations on
#     Apple Metal Performance Shaders (MPS) - a framework for GPU-accelerated machine learning on Apple Silicon.
#     It ensures that the model training and other computations are correctly configured to run on MPS-capable devices.
#
# Methods:
#   - `setup_device`: 
#       Sets up the specific MPS device for the training process. If a non-MPS device is provided, 
#       it raises a `MisconfigurationException` to ensure the use of MPS.
#
#   - `get_device_stats`: 
#       Provides a dictionary containing statistics and information about the MPS device by calling 
#       an internal `get_device_stats()` function (which would typically gather GPU utilization and memory stats).
#
#   - `teardown`: 
#       A placeholder for any cleanup operations that need to be performed post training on the MPS device.
#
#   - `parse_devices`: 
#       Interprets device descriptors provided by the user to configure a list of MPS devices for training.
#
#   - `register_accelerators`: 
#       Registers the `MPSAccelerator` within the given `accelerator_registry`, making it discoverable and usable 
#       within the FlameModules ecosystem.
#
#   - `auto_device_count`:
#       A static method that returns the number of available MPS devices (commonly 1 on Apple devices).
#
#   - `is_available`:
#       Checks if the MPS accelerator is available on the current machine.
#
# Parameters:
#   - `device`: An instance of `torch.device` indicating the type of device to be set up for MPS.
#   - `devices`: A descriptor that can be an integer, string, or a list of integers, which is parsed to configure MPS devices.
#   - `accelerator_registry`: A registry object to which the MPSAccelerator class adds itself.
#
# Returns:
#   - For `get_device_stats`: Returns a dictionary with MPS device statistics.
#   - For `parse_devices`: Returns a list of torch.device objects configured for MPS.
#
# Key Steps:
#   1. `setup_device` verifies and prepares the MPS device for model operations.
#   2. `get_device_stats` pulls in device statistics for monitoring or optimization purposes.
#   3. `teardown` is ready to handle any post-processing after training.
#   4. `parse_devices` transforms user input into a usable MPS device configuration.
#   5. `register_accelerators` introduces the MPSAccelerator to the system for selection and use.
#   6. `auto_device_count` informs about the count of MPS-capable devices, aiding in resource allocation.
#   7. `is_available` serves as a check before attempting to use the MPS device, ensuring compatibility.
#
# Importance:
#   - This class is vital for leveraging the GPU capabilities of Apple Silicon in neural network training, offering
#     efficient computation and energy usage.
#   - It provides interfaces for MPS device management, promoting modularity and adaptation within the FlameModules
#     framework for Apple hardware environments.


#@: Apple Metal Silicon GPU Devices
class MPSAccelerator(Accelerator):
    
    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != 'mps':
            raise MisconfigurationException(f'Device should be MPS, got {device} instead.')
    
    
    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        return get_device_stats()
    
    
    
    @override
    def teardown(self) -> None:
        ...
        
        
    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> list[torch.device]:
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None
        return [
            torch.device('mps', idx) for idx in range(len(parsed_devices))
        ]



    @staticmethod
    @override
    def auto_device_count() -> int:
        return 1
    
    
    
    @staticmethod
    @override
    def is_available() -> bool:
        return _MPSAccelerator.is_available()
    
    
    
    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            'mps', cls, description= cls.__name__
        )




#@: Device metrics
_vm_percent: str = 'm3_vm_percent'
_percent: str = 'm3_percent'
_swap_percent: str = 'm3_swap_percent'



def get_device_stats() -> dict[str, float]:
    if not _psutil_available:
        raise ModuleNotFoundError(f'Fetching MPS device stats requires `psutil` to be installed. {str(_psutil_available)}')

    import psutil
    
    return {
        _vm_percent: psutil.virtual_memory().percent,
        _percent: psutil.cpu_percent(),
        _swap_percent: psutil.swap_memory().percent
    }
    
    
