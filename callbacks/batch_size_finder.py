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
from lightning.pytorch.tuner.batch_size_scaling import _scale_batch_size
from lightning.pytorch.utilities.exceptions import MisconfigurationException, _TunerExitException
from lightning.pytorch.utilities.parsing import lightning_hasattr
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

# ----------------------------------
# BatchSizeFinder in FlameModules
# ----------------------------------
# What is BatchSizeFinder?
#
#     BatchSizeFinder is a callback utility used in FlameModules, a hypothetical deep learning framework. 
#     Its primary function is to automatically determine the optimal batch size for training a neural 
#     network model on a given hardware setup. Instead of manually experimenting with various batch sizes 
#     and monitoring for potential 'Out Of Memory (OOM)' errors, BatchSizeFinder efficiently scales up 
#     the batch size, ensuring that the highest feasible size is identified without causing a memory overflow.
#
# Why do we need BatchSizeFinder in FlameModules?
#
#     1. Optimize Training Speed: By utilizing the right batch size, training can be significantly faster. 
#        Too small a batch size may not fully utilize the GPU's capabilities, while too large might cause 
#        memory errors. BatchSizeFinder helps find a balance that maximizes throughput.
#
#     2. Automated Resource Management: Manually tuning the batch size to avoid OOM errors can be tedious. 
#        BatchSizeFinder automates this process, saving time and ensuring optimal resource utilization.
#
#     3. Consistency in Experiments: When experimenting with different models or datasets, having a consistent 
#        methodology to determine the best batch size ensures that results are comparable and consistent.
#
#     4. Enhance Productivity: Developers or researchers no longer need to babysit the training process, 
#        adjusting batch sizes based on potential failures. BatchSizeFinder takes care of this, allowing 
#        them to focus on other important tasks.
#
#     5. Flexibility & Scalability: As hardware environments change, with different GPU memory availabilities, 
#        the ideal batch size might differ. BatchSizeFinder provides a flexible approach that can adapt to 
#        different hardware configurations, making the code more scalable.
#
# How to use BatchSizeFinder in FlameModules?
#
#     Typically, BatchSizeFinder is initialized with certain parameters, such as mode and initial batch size. 
#     Once initialized, it can be passed as a callback to the FlameModule's training routine. During training, 
#     it will dynamically adjust and test various batch sizes to find the optimal one without manual intervention.

class BatchSizeFinder(Callback):
    # The `BatchSizeFinder` callback tries to find the largest batch size for a given model that does not
    # give an `Out Of Memory (OOM)` error. 
    # Internally it calls the respective step function `steps_per_trial` times for each batch size until one of 
    # the batch sizes generates an OOM error.
    
    # Code Docx:
    #       import FlameModules as flame_modules
    #       from flame_modules.callbacks.batch_size_finder import BatchSizeFinder
    #       
    #
    #       class EvalBatchSizeFinder(BatchSizeFinder):
    #           def __init__(self, *args: list[Any], **kwargs: dict[Any, Any]) -> None:
    #               super(EvalBatchSizeFinder, self).__init__()
    #          
    #           def on_fit_start(self, *args: list[Any], **kwargs: dict[Any, Any]) -> ...:
    #               return ...
    #
    #           def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
    #               self.scale_batch_size(trainer, flame_module)
    #
    #
    #       if __name__.__contains__('__main__'):
    #           trainer = Trainer(callbacks= [EvalBatchSizeFinder()])
    #           trainer.test(...)
    
    supported_modes: tuple[str] = ('power', 'binsearch')
    
    
    def __init__(self, model: Optional[str] = 'power', steps_per_trial: Optional[int] = 3, init_val: Optional[int] = 2, max_trials: Optional[int] = 25, 
                                                                                                                        batch_arg_name: Optional[str] = 'batch_size') -> None:
        # Purpose:
        #     Initializes the BatchSizeFinder with specified parameters to guide batch size optimization.
        #
        # Parameters:
        #   - `model`: The method used to search for optimal batch size. Defaults to 'power'.
        #   - `steps_per_trial`: Number of steps to run for each batch size. Defaults to 3.
        #   - `init_val`: Initial batch size for the search. Defaults to 2.
        #   - `max_trials`: Maximum trials to perform during search. Defaults to 25.
        #   - `batch_arg_name`: Argument name for adjusting batch size. Defaults to 'batch_size'.
        #
        # Key Steps:
        #   1. Validate if the provided `model` is within the supported modes.
        #   2. Initialize instance variables using the provided parameters.
        
        mode: str = mode.lower()
        if mode not in self.supported_modes:
            raise ValueError(f'`mode` should be either of {self.supported_modes}')
        
        self.optional_batch_size: Optional[int] = init_val
        self._mode = mode
        self._steps_per_trial = steps_per_trial
        self._init_val = init_val
        self._max_trials = max_trials
        self._batch_arg_name = batch_arg_name
        self._early_exit = False
        
        
        
        
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: Optional[str] = None) -> None:
        # Purpose:
        #     Prepares the training environment for the batch size search, ensuring compatibility and configuration correctness.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #   - `stage`: The stage in the training process, default is 'fit'.
        #
        # Key Steps:
        #   1. Validate if distributed strategies are not used.
        #   2. Ensure dataloaders are passed through the FlameModule or LightningDataModule, not directly.
        #   3. Confirm that only one dataloader is used during the specified stage.
        #   4. Check if the batch size argument is present in the module or its parameters.
        #   5. Warn if duplicate batch size settings are detected in both module and its parameters.
        if trainer._accelerator_conector.is_distributed:
            raise MisconfigurationException('The Batch size finder is not supported with distributed strategies.')
    
        if not trainer.fit_loop._data_source._is_module():
            raise MisconfigurationException(
                "The Batch size finder cannot be used with dataloaders passed directly to `.fit()`. Please disable"
                " the feature or incorporate the dataloader into your LightningModule or LightningDataModule."
            )
            
        if stage != 'fit':
            loop = trainer._active_loop
            assert loop is not None
            loop.setup_data()
            combined_loader = loop._combined_loader
            assert combined_loader is not None
            if len(combined_loader.flattened) > 1:
                stage = trainer.state.stage
                assert stage is not None
                raise MisconfigurationException(
                    f"The Batch size finder cannot be used with multiple {stage.dataloader_prefix} dataloaders."
                )
                
        
        if not lightning_hasattr(flame_module, self._batch_arg_name):
            raise MisconfigurationException(
                f"Field {self._batch_arg_name} not found in `model`, `datamodule`, nor their `hparams` attributes."
            )
            
        
        if (hasattr(flame_module, self._batch_arg_name) and hasattr(flame_module, 'hparams') and self._batch_arg_name in flame_module.hparams):
            rank_zero_warn(
                f"Field `model.{self._batch_arg_name}` and `model.hparams.{self._batch_arg_name}` are mutually"
                f" exclusive! `model.{self._batch_arg_name}` will be used as the initial batch size for scaling."
                " If this is not the intended behavior, please remove either one."
            ) 
            
        
        
    def scale_batch_size(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     Adjusts the batch size based on the specified method and updates the optimal batch size.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #
        # Key Steps:
        #   1. Determine the new batch size using the `_scale_batch_size` function.
        #   2. Update the `optimal_batch_size` attribute with the newly determined batch size.
        #   3. If the `_early_exit` flag is set, raise an exception to terminate the tuner early.
        new_size = _scale_batch_size(
            trainer, self._mode, self._steps_per_trial, self._init_val, self._max_trials, self._batch_arg_name
        )
        
        self.optimal_batch_size = new_size
        if self._early_exit:
            raise _TunerExitException()
        
        
    
    def on_fit_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     Calls the batch size scaling method when the fitting process starts.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #
        # Key Steps:
        #   1. Initiate the batch size scaling process by calling `scale_batch_size` method.
        self.scale_batch_size(trainer, flame_module)
        
    
    
    
    def on_validation_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     Initiates batch size scaling at the start of the validation phase.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #
        # Key Steps:
        #   1. Skip the scaling process during sanity checks or non-validation functions.
        #   2. Initiate batch size scaling for the validation phase.
        if trainer.sanity_checking or trainer.state.fn != 'validate':
            return 
        
        self.scale_batch_size(trainer, flame_module)
        
        
        
    def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     Initiates batch size scaling at the start of the testing phase.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #
        # Key Steps:
        #   1. Initiate batch size scaling for the testing phase.
        self.scale_batch_size(trainer, flame_module)
        
        
    
    def on_predict_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #      Initiates batch size scaling at the start of the prediction phase.
        #
        # Parameters:
        #   - `trainer`: Instance of `flame_modules.Trainer`.
        #   - `flame_module`: Model instance, `flame_modules.FlameModule`.
        #
        # Key Steps:
        #   1. Initiate batch size scaling for the prediction phase.
        self.scale_batch_size(trainer, flame_module)
        
        
