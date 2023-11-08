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
import torch, logging, os, shutil, subprocess
from typing_extensions import override
from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.cuda import _check_cuda_matmul_precision, _clear_cuda_memory, num_cuda_devices
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from FlameModules.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


_log = logging.getLogger(__name__)


# --------------------------------
# CUDAAccelerator in FlameModules
# --------------------------------

# Purpose:
#     The `CUDAAccelerator` class extends the `Accelerator` base class for CUDA-enabled GPU operations
#     within the FlameModules framework. It ensures models are set up for training on CUDA devices, 
#     managing device allocation and statistics specifically for NVIDIA GPUs.
#
# Methods:
#   - `setup_device`: 
#       Ensures that the provided device is a CUDA device and sets the current device accordingly.
#       If a non-CUDA device is passed, it raises a `MisconfigurationException`.
#
#   - `setup`: 
#       Sets NVIDIA-specific flags and clears CUDA memory to prepare for training.
#
#   - `set_nvidia_flags`: 
#       A static method that sets the `CUDA_DEVICE_ORDER` environment variable and logs the 
#       visible CUDA devices for the current local rank.
#
#   - `get_device_stats`: 
#       Fetches and returns a dictionary of CUDA memory statistics for the given device.
#
#   - `teardown`: 
#       A method to clear CUDA memory after the training process has finished.
#
#   - `parse_devices`: 
#       Interprets the provided device descriptors and returns a list of CUDA device IDs if available.
#
#   - `get_parallel_devices`: 
#       Translates a list of device IDs into a list of `torch.device` objects for parallel GPU training.
#
#   - `auto_device_count`: 
#       Returns the number of available CUDA devices for the current environment.
#
#   - `is_available`: 
#       Checks and returns a boolean indicating whether any CUDA devices are available.
#
#   - `register_accelerators`: 
#       Registers the `CUDAAccelerator` within the provided `accelerator_registry` to make it 
#       available for use within FlameModules.
#
# Parameters:
#   - `device`: An instance of `torch.device` indicating the CUDA device to be configured.
#   - `trainer`: An instance of `flame_modules.Trainer` that holds training configurations and state.
#   - `local_rank`: The local rank of the GPU device in distributed training scenarios.
#   - `devices`: Can be an integer, a string, or a list of integers specifying CUDA device IDs.
#   - `accelerator_registry`: A registry where the `CUDAAccelerator` is registered for later use.
#
# Returns:
#   - `get_device_stats`: Returns a dictionary with CUDA memory statistics for the specified device.
#   - `parse_devices`: Returns a list of CUDA device IDs based on the input format.
#   - `get_parallel_devices`: Returns a list of `torch.device` objects for the specified CUDA devices.
#   - `auto_device_count`: Returns the count of available CUDA devices.
#   - `is_available`: Returns True if CUDA devices are available, otherwise False.
#
# Key Steps:
#   1. The `setup_device` method validates and configures the CUDA device for the model.
#   2. The `setup` method optimizes CUDA-specific settings for the training session.
#   3. The `get_device_stats` method provides insight into the GPU's memory usage for monitoring.
#   4. The `teardown` method ensures the cleanup of CUDA memory post-training.
#   5. The `parse_devices` method allows correct identification and formatting of CUDA device IDs.
#   6. The `get_parallel_devices` method prepares the GPU devices for distributed training.
#   7. The `register_accelerators` method incorporates the `CUDAAccelerator` into the FlameModules ecosystem.
#
# Importance:
#   - This class is essential for harnessing the full power of NVIDIA GPUs, facilitating high-performance 
#     model training through efficient resource management and setup.
#   - It also helps in maintaining a clean and effective training environment with proper device initialization 
#     and teardown procedures.
#
class CUDAAccelerator(Accelerator):
    
    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != 'cuda':
            raise MisconfigurationException(f'Device whould be on GPU, got {device} instead.')
        _check_cuda_matmul_precision(device)
        torch.cuda.set_device(device)
        
        
    @override
    def setup(self, trainer: 'flame_modules.Trainer') -> None:
        self.set_nvidia_flags(trainer.local_rank)
        _clear_cuda_memory()
        
        
    
    @staticmethod
    def set_nvidia_flags(local_rank: int) -> None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        all_gpu_ids = ','.join(str(x) for x in range(num_cuda_devices()))
        devices = os.getenv('CUDA_VISIBLE_DEVICES', all_gpu_ids)
        _log.info(f'LOCAL RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]')
        
        
        
        
    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        return torch.cuda.memory_stats(device)
    
    
    
    @override
    def teardown(self) -> None:
        _clear_cuda_memory()
        
        
    
    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        return _parse_gpu_ids(devices, include_cuda= True)
    
    
    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device('cuda', x) for x in devices]
    
    
    
    @staticmethod
    @override
    def auto_device_count() -> int:
        return num_cuda_devices()
    
    
    
    @staticmethod
    @override
    def is_available() -> bool:
        return num_cuda_devices() > 0
    
    
    
    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register('cuda', cls, description= cls.__name__)
        
        


def get_nvidia_gpu_stats(device: _DEVICE) -> dict[str, float]:
    nvidia_smi_path = shutil.which('nvidia-smi')
    if nvidia_smi_path is None:
        raise FileNotFoundError('nvidia-smi: command not found')
    
    gpu_stat_metrics: list[tuple[str, str]] = [
        ('utilization.gpu', '%'),
        ('memory.used', 'MB'),
        ('memory.free', 'MB'),
        ('utilization.memory', '%'),
        ('fan.speed', '%'),
        ('temperature.gpu', '°C'),
        ('temperature.memory', '°C')
    ]
    gpu_stat_keys: list[str] = [key for key, _ in gpu_stat_metrics]
    gpu_query = ','.join(gpu_stat_keys)
    
    index = torch._utils._get_device_index(device)
    gpu_id = _get_gpu_id(index)
    result = subprocess.run(
        [nvidia_smi_path, f'--query-gpu={gpu_query}', '--format= csv, nounits, noheader', f'--id= {gpu_id}'],
        encoding= 'utf-8',
        capture_output= True,
        check= True
    )
    
    def _to_float(x: str) -> float:
        try: return float(x)
        except ValueError: return 0.0
        
    s = result.stdout.strip()
    stats = [_to_float(x) for x in s.split(', ')]
    return {
        f'{x} ({unit})': stat for (x, unit), stat in zip(gpu_stat_metrics, stats)
    }
    
    
    
def _get_gpu_id(device_id: int) -> str:
    default = ','.join(str(i) for i in range(num_cuda_devices()))
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', default= default).split(',')
    return cuda_visible_devices[device_id].strip()
        
        
        
        
        
