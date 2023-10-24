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

__note: str = r'''
    In PyTorch, and in deep learning terminology in general, "sample" and "batch" are terms used to describe subsets of your dataset. 
    Here's the differentiation:

    *  Sample:
        * A single element of your dataset.
        * For example, in a dataset of images, a single image and its corresponding label or target would be one sample.


    *  Batch:
        * A group of samples.
        * The number of samples in a batch is defined by the batch size, which is a hyperparameter you choose.

    In PyTorch, when you use a DataLoader with a defined batch_size, it loads data in batches. Each batch contains multiple samples 
    up to the specified batch size.
'''

#@: Data Hooks
# Hooks to be used for data related stuff.
class Datahooks:
    def prepare_data(self) -> None:
        # Use this to download and prepare data
        #@: NOTE: Will change later 
        ...
        
    def setup(self, stage: str) -> None:
        # Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when you
        # need to build models dynamically or adjust something about them. This hook is called on every process when
        # using DDP.

        # Args:
        #     stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``      
        ...
        
    def teardown(self, stage: str) -> None:
        # Called at the end of fit (train + validate), validate, test, or predict.

        # Args:
        #     stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        ...
        
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        ...
        
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        ...
        
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        ...
        
        
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        ...
        
        
    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_index: int) -> Any:
        # sample: dict[str, Any] = {
        #     'independent_variable': x_value,
        #     'dependent_variable': y_value
        # }
        
        # batch: list[dict[str, Any]] = { list of 'n' sample }
        for sample in batch:
            sample.independent_variable: Any = sample.independent_variable.to(device)
            sample.dependent_variable: Any = sample.dependent_variable.to(device)
        
        return batch    


    def on_before_batch_transfer(self, batch: Any, dataloader_index: int) -> Any:
        # Override to alter or apply batch augmentations to your batch before it is transferred to the device.
        return batch
    
    
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # Override to alter or apply batch augmentations to your batch after it is transferred to the device.
        return batch
    

        

#@: Model hooks 
# This code defines a class named Modelhooks, which provides hooks or callback points 
# for various stages in the lifecycle of a model's training, validation, and testing.
class Modelhooks:    
    def on_fit_start(self) -> None:
        # This method is called at the beginning of the entire fitting process.
        # If on DDP it is called on every process
        ...
        
    def on_fit_end(self) -> None:
        # Called at the very end of fit.
        # If on DDP it is called on every proces
        ...
    
    def on_train_start(self) -> None:
        # Called at the beginning of training after sanity check.
        ...
        
    def on_train_end(self) -> None:
        # Called at the end of training before logger experiment is closed.
        ...
        
    def on_validation_start(self) -> None:
        # Called at the beginning of validation.
        ...
        
    def on_validation_end(self) -> None:
        # Called at the end of validation.
        ...
    
    def on_test_start(self) -> None:
        # Called at the beginning of testing
        ...
        
    def on_test_end(self) -> None:
        # Called at the end of testing.
        ...
        
    def on_predict_start(self) -> None:
        # Called at the beginning of predicting
        ...
        
    def on_predict_end(self) -> None:
        # Called at the end of predicting.
        ...
        
    def on_train_batch_start(self, batch: Any, batch_index: int) -> Optional[int]:
        # Called in the training loop before anything happens for that batch.
        ...
    
    def on_train_batch_end(self, outputs: 'step_output', batch: Any, batch_index: int) -> None:
        # outputs: The outputs of training_step(x)
        ...
    
    
    def on_validation_batch_start(self, batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
        
    
    def on_validation_batch_end(self, outputs: 'step_output', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
        
        
    def on_test_batch_start(self, batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
        
    
    def on_test_batch_end(self, outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
    
    
    def on_predict_batch_start(self, batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
        
        
    def on_predict_batch_end(self, outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        ...
        
        
    
    def on_validation_model_zero_grad(self) -> None:
        #@: Called by the training loop to release gradients before entering the validation loop.
        # zero_grad_kwargs = {} if _TORCH_GREATER_EQUAL_2_0 else {"set_to_none": True}
        # self.zero_grad(**zero_grad_kwargs)
        
        ...
        
        
    def on_validation_model_eval(self) -> None:
        #@: set the model to eval mode during the validation loop
        self.trainer.model.eval()
        
        
    def on_validation_model_train(self) -> None:
        #@: set the model to train during the validation loop
        self.trainer.model.train()
 
 
    def on_test_model_train(self) -> None:
        #@: set the model to train during the test loop
        self.trainer.model.train()
    
    
    def on_test_model_eval(self) -> None:
        #@: set the model to eval during the test loop
        self.trainer.model.eval()
        
        
    def on_predict_model_eval(self) -> None:
        #@: set the model to eval during the predict loop
        self.trainer.model.eval()
        
        
    def on_train_epoch_start(self) -> None:
        #@: called in the training loop at the very beginning of the epoch
        ...
        
        
    def on_train_epoch_end(self) -> None:
        #@: called in the training loop at the very end of the epoch.
        
        # class MyModule({Main_Module}):
        #     def __init__(self):
        #         super().__init__()
        #         self.training_step_outputs = []

        #     def training_step(self):
        #         loss = ...
        #         self.training_step_outputs.append(loss)
        #         return loss

        #     def on_train_epoch_end(self):
        #         # do something with all training_step outputs, for example:
        #         epoch_mean = torch.stack(self.training_step_outputs).mean()
        #         self.log("training_epoch_mean", epoch_mean)
        #         # free up the memory
        #         self.training_step_outputs.clear()    
        ...
    
    
    def on_validation_epoch_start(self) -> None:
        #@: Called in the validation loop at the very beginning of the epoch.
        ...
        
    
    def on_validation_epoch_end(self) -> None:
        #@: Called in the validation loop at the very end of the epoch.
        ...
        
        
    def on_test_epoch_start(self) -> None:
        #@: Called in the test loop at the very beginning of the epoch.
        ...
        
        
    def on_train_epoch_end(self) -> None:
        #@: Called in the test loop at the very end of the epoch.
        ...
        
        
    def on_predict_epoch_start(self) -> None:
        #@: Called at the beginning of predicting
        ...
    
    def on_predict_epoch_end(self) -> None:
        #@: Called at the end of predicting 
        
        ...
        
        
    def on_before_zero_grad(self, optimizer: 'torch.optim.Optimizer') -> None:
        #@: Called after ''training_step()'' and before ''optimizer.zero_grad()''
        # Called in the training loop after taking an optimizer step and before zeroing grads.
        # Good place to inspect weight information with weights updated.

        # This is where it is called := 

        #     for optimizer in optimizers:
        #         out = training_step(...)

        #         model.on_before_zero_grad(optimizer) # < ---- called here
        #         optimizer.zero_grad()

        #         backward()
        ...
        
        
    def on_before_backward(self, loss: torch.tensor) -> None:
        #@: Called before ''loss.backward()''
        #@: ARGS: loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.
        ...
        
    def on_after_backward(self) -> None:
        #@: Called after ''loss.backward()'' and before optimizers are stepped.
        ...
        
    
    def on_before_optimizer_step(self, optimizer: 'torch.optim.Optimizer') -> None:
        #@: Called before ''optimizer.step()''
        ...
        
        
    def configure_model(self) -> None:
        # Hook to create modules in a strategy and precision aware context.

        # This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
        # the model instantly to save memory and initialization time.
        # For non-sharded strategies, you can choose to override this hook or to initialize your model under the
        

        # This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
        # implementation of this hook is idempotent.
        ...
        
    
    
    
class Checkpointhooks:
    # Hooks to be used with Checkpointing.
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        ...
        
        
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        ...     


#@: Driver Code
if __name__.__contains__('__main__'):
    print('hemllo')
    
    
    
    
