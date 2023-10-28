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

# -----------------------------------
# Understanding Callbacks in AI
# -----------------------------------
#
# Callbacks: A Definition
# -----------------------
# In the context of machine learning and deep learning, a callback is a design pattern that 
# allows custom functionalities to be injected into existing processes. Specifically, a callback 
# refers to a set of functions that can be registered and are executed at specified stages 
# during the training, evaluation, or inference of a model. 
#
# Purpose and Utility of Callbacks
# --------------------------------
# Callbacks serve multiple purposes in the training and evaluation loop:
#       1. Adaptability: They offer a flexible mechanism to extend or modify the default behavior 
#                        of training loops without modifying their underlying code.
#       
#       2. Monitoring: Callbacks allow for real-time monitoring of any metric or value, ensuring 
#                      that the training process is transparent and under control.
#       
#       3. Control: They offer the capability to halt or alter the training process based on custom 
#                   conditions or triggers.
#       
#       4. Automation: Callbacks can automate repetitive tasks like saving model checkpoints or 
#                      adjusting hyperparameters.
#
# Key Types of Callbacks in AI
# ----------------------------
# While the specific callbacks can vary based on the framework or library in use, several common 
# callback archetypes exist:
#       1. Monitoring Callback: Observes specific metrics, logs them, and potentially visualizes 
#                               them in tools like TensorBoard.
# 
#       2. Early Stopping Callback: Monitors a specified metric (e.g., validation loss) and stops 
#                                   the training process if the metric does not improve over a 
#                                   defined number of epochs.
# 
#       3. Model Checkpointing: Saves model weights and architecture periodically or when a specific 
#                               condition is met.
# 
#       4. Learning Rate Scheduling: Dynamically adjusts the learning rate based on predefined rules 
#                                    or performance metrics.
# 
#       5. Custom Function Callback: Allows any user-defined function to be called at specific 
#                                    stages, offering almost unlimited customization.
# 
#       6. Regularization Callbacks: Implements techniques such as gradient clipping or custom weight 
#                                    updates during training.
#
# Summary of Callbacks
# ---------------------------
# Callbacks are indispensable tools that significantly enhance the transparency, control, and 
# flexibility of model training and evaluation processes. However, care should be taken while 
# using callbacks. Too many callbacks can clutter and complicate the training loop. It's essential 
# to ensure that the interaction of multiple callbacks doesn't produce unintended behaviors. As 
# with all tools, understanding and judicious use of callbacks lead to the most effective results.


from __future__ import annotations
import torch
from typing import Type


# class Callback:
#   - This is an abstract base class intended to be subclassed when creating new callbacks.

#   @property
#   def state_key(self) -> str:
#       - Provides an identifier for the state of the callback.
#       - The purpose of this method is to return a unique identifier that will be used to store and retrieve 
#         the state of a callback from the checkpoint dictionary. Specifically, it will be stored under
#         `checkpoint['callbacks'][state_key]`.
#       - Callback implementations need to ensure that the returned state key is unique, especially if:
#           1. The callback maintains some internal state.
#           2. There is a requirement to save the state of multiple instances of that callback.
#       - By default, it returns the qualified name of the class.

#   @property
#   def _legacy_state_key(self) -> Type['Callback']:
#       - Returns the type of the current instance (i.e., the class of the callback).
#       - This is presumably for compatibility or transition purposes, hence the term "legacy".

#   def _generate_state_key(self, **kwargs: Any) -> str:
#       - This is a helper method to format a set of key-value pairs into a state key string. 
#       - The resulting string has the callback class name as a prefix.
#       - This can be particularly useful when defining the `state_key`.
#       - Args:
#           1. **kwargs: A set of key-value pairs. These should be serializable to a string.

#   def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
#       - This method is called at the start of operations like fit, validate, test, predict, or tune.
#       - It provides an opportunity to initialize or set up necessary variables or structures 
#         before the main operation begins.

#   def teardown(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
#       - This method is called at the end of operations like fit, validate, test, predict, or tune.
#       - Can be used for cleanup, finalization, or any other wrap-up tasks post-operation.

# def on_fit_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - This method is called at the start of the fit operation.
#   - Can be used to set up any necessary preparations specific to the fit operation.

# def on_fit_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - This method is called at the end of the fit operation.
#   - Can be used for any wrap-up or finalization tasks specific to the fit operation.

# def on_sanity_check_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called when the validation sanity check begins.
#   - Provides an opportunity to run any initial setup or checks before validation starts.

# def on_sanity_check_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called after the validation sanity check is complete.
#   - Can be used for tasks like cleanup or final checks after validation.

# def on_train_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int) -> None:
#   - This method is called at the beginning of each training batch.
#   - Provides an opportunity to set up or preprocess anything needed before processing the batch.
#   - The current batch and its index are provided as arguments.

# def on_train_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_output', batch: Any, batch_index: int) -> None:
#   - Called at the end of each training batch.
#   - Can be used for tasks like post-processing, saving intermediate results, or any other post-batch operations.
#   - NOTE: The value of `outputs['loss']` at this point will be normalized with respect to the `accumulate_grad_batches` setting. 
#     This value represents the loss value returned from the `training_step`.

# def on_train_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the beginning of a training epoch.
#   - Can be used to initialize or set up any resources needed for the upcoming epoch.

# def on_train_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the end of a training epoch.
#   - Can be used to perform operations like calculating and logging the mean of all the outputs from training steps of the epoch.
#   - Example Use Case:
#     - A custom FlameModule might accumulate outputs from each training step.
#     - A custom callback can then use this method to calculate the mean of these outputs for the epoch and then log it.
#     - Finally, the accumulated outputs can be cleared to free up memory.

# def on_validation_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the beginning of a validation epoch.
#   - Can be used to set up or initialize anything specific to validation.

# def on_validation_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the end of a validation epoch.
#   - Can be used to finalize, log, or perform other tasks specific to the validation phase.

# def on_test_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the beginning of a testing epoch.
#   - Can be utilized for setup or initial tasks required for testing.

# def on_test_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Called at the end of a testing epoch.
#   - Suitable for wrapping up tasks or logging results related to the testing phase.

# def on_predict_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Initiated at the start of a prediction epoch.
#   - Useful for any preparations or setups specifically tied to the prediction phase.

# def on_predict_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Invoked at the close of a prediction epoch.
#   - Ideal for any concluding tasks or finalizing actions associated with the prediction phase.

# def on_validation_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Invoked at the beginning of processing a validation batch.
#   - Can be utilized to prepare any resources or state specific to the current validation batch.

# def on_validation_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Called post-processing a validation batch.
#   - Useful for wrapping up tasks or logging results associated with the specific validation batch.

# def on_test_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Initiated at the start of a testing batch.
#   - Ideal for any preparations or initial tasks specific to the test batch.

# def on_test_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Concludes upon the end of a testing batch.
#   - Suited for finalizing tasks, analyzing outputs, or logging results specific to the testing batch.

# def on_predict_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Triggered at the outset of a prediction batch.
#   - Can be harnessed to arrange or initiate any specific logic for the upcoming prediction batch.

# def on_predict_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
#   - Ends at the conclusion of processing a prediction batch.
#   - Effective for final tasks, result analysis, or logging tied to the prediction batch.

# def on_train_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Commences when the training process is about to start.
#   - Useful for setting up resources, initializing states, or any other preparations before training begins.

# def on_train_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Marks the end of the training process.
#   - Suited for cleanup, finalizing results, or other concluding operations post-training.

# def on_validation_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Activates at the beginning of the validation loop.
#   - Can be employed for preparations or setups particularly related to the validation loop.

# def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Signs off at the conclusion of the validation loop.
#   - Useful for wrap-up activities or logging results post-validation.

# def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Activates as the testing process initiates.
#   - Utilized for setup, initializing resources, or any initial tasks before the test starts.

# def on_test_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Commences upon the conclusion of the testing process.
#   - Ideal for finalizing tasks, logging results, or cleanup activities post-test.

# def on_predict_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Kicks in at the beginning of the prediction phase.
#   - Useful for setting up resources, initializing any states, or preliminary tasks before predictions begin.

# def on_predict_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Marks the end of the prediction process.
#   - Can be harnessed for post-prediction wrap-up activities, result logging, or other concluding operations.

# def on_exception(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', exception: BaseException) -> None:
#   - Activates when the trainer's execution gets interrupted by an exception.
#   - Beneficial for exception handling, logging errors, or executing any remediation steps if necessary.

# def state_dict(self) -> dict[str, Any]:
#   - Invoked during the checkpoint saving process.
#   - Implements the logic to generate a callback's `state_dict`.
#   - Returns:
#     - A dictionary encapsulating the state of the callback.

# def load_state_dict(self, state_dict: dict[str, Any]) -> None:
#   - Triggered during the checkpoint loading phase.
#   - Implements the logic to reload the callback's state from a provided `state_dict`.
#   - Args:
#     - state_dict: The dictionary representing the callback's state, as returned by `state_dict`.

# def on_save_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
#   - Activates during the checkpoint saving procedure.
#   - Allows additional data to be included in the checkpoint if necessary. For example, any specific states, configurations, or other data required for restoring the model later.

# def on_load_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
#   - Initiates during the model checkpoint loading process.
#   - Helpful for restoring any specific states, configurations, or other data that was stored during checkpointing.

# def on_before_backward(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', loss: torch.tensor) -> None:
#   - Activates just before the backward pass is performed with `loss.backward()`.
#   - Ideal for operations that need to occur prior to the backward pass, like modifying gradients, setting hooks, or other preliminary tasks.

# def on_after_backward(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#   - Commences right after the backward pass with `loss.backward()` and before optimizer updates.
#   - Useful for post-backward operations like gradient clipping, logging gradients, or any other tasks before the optimizer steps in.

# def on_before_optimizer_step(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', optimizer: torch.optim.Optimizer) -> None:
#   - Triggers just before the optimizer updates model parameters using `optimizer.step()`.
#   - Useful for any preliminary tasks or configurations before the optimizer starts its update.

# def on_before_zero_grad(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', optimizer: torch.optim.Optimizer) -> None:
#   - Kicks in right before gradients are set to zero with `optimizer.zero_grad()`.
#   - Helpful for operations that need to be done before resetting gradients, like logging old gradients or other related tasks.

class Callback:
    #@: Abstract base class for building new callbacks
    #@: Subclass this class and then override any of the relevant hooks
    @property
    def state_key(self) -> str:
        #@: Identifier for the state of the callback
        #@: Used to store and retreive a callback's state from the checkpoint dictionary by
        # `checkpoint['callbacks'][state_key]`. Implementations of a callback need to provide 
        # unique state key if : 
        #   1. The callback has state
        #   2. It is desired to maintain the state opf multiple instances of that callback.
        return self.__class__.__qualname__
    
    
    @property
    def _legacy_state_key(self) -> Type['Callbacl']:
        return type(self)
    
    
    def _generate_state_key(self, **kwargs: Any) -> str:
        # Formats a set of key-value pairs into a state key string with the callback class name prefixed.
        # Useful for defining a :attr: `state_key`
        # Args:
        #   1. **kwargs: set of key-value pairs. Must be serializable to :class: `str`
        return f'{self.__class__.__qualname__}{repr(kwargs)}'
    
    
    
    def setup(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        #@: Called when fit, validate, test, predict, or tune begins. 
        ...
        
    
    def teardown(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', stage: str) -> None:
        #@: Called when fit, validate, test, predict, or tune ends.
        ...
        
    
    def on_fit_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when fit begins
        ...
        
    
    def on_fit_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when fit ends
        ...
        
        
    def on_sanity_check_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the validation sanity check starts.
        ...
        
        
    def on_sanity_check_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the validation sanity check ends. 
        ...
        
    
    def on_train_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int) -> None:
        #@: Called then the train batch begins. 
        ...
        
    
    def on_train_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_output', batch: Any, batch_index: int) -> None:
        #@: Called when the train batch ends. 
        # NOTE :
        #   The Value of `outputs['loss']` here will be the normalized value w.r.t `accumulate_grad_batches` of the loss returned from `training_step`
        ...
        
    def on_train_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called then the train epoch begins.
        ...
        
        
    def on_train_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the train epoch ends. 
        # class NewFlameModule(flame_modules.FlameModule):
        #     def __init__(self) -> None:
        #         super(NewFlameModule, self).__init__()
        #         self.training_step_outputs: list[torch.tensor] = []
        #        
        #
        #     def training_step(self) -> torch.tensor:
        #         loss: torch.tensor = ...
        #         self.training_step_outputs.append(loss)
        #         return loss
        #   
        #
        # class MyFlameCallback(flame_modules.Callback):
        #     def on_train_epoch_end(self, trainer: ..., flame_module: ...) -> None:
        #         # do something with all the training_step outputs
        #         # Eg.
        #         epoch_mean: torch.tensor = torch.stack(
        #             flame_module.training_step_outputs
        #         ).mean()
        #         flame_module.log('training_epoch_mean', epoch_mean)
        #         #@: free up the memory
        #         flame_module.training_step_outputs.clear()
        ...
        
        
    def on_validation_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `val epoch` begins. 
        ...
        
        
    def on_validation_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `val epoch` ends.
        ...
        
    
    def on_test_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `test epoch` begins.
        ...
        
        
    def on_test_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `test epoch` ends.
        ...
        
        
    def on_predict_epoch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `predict epoch` begins.
        ...
        
        
    def on_predict_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the `predict epoch` ends.
        ...
        
        
    def on_validation_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `validation batch` begins.
        ...
        
    def on_validation_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `validation batch` ends. 
        ...
    
    def on_test_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `test batch` begins.
        ...
        
    def on_test_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `test batch` ends. 
        ...
        
    def on_predict_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `predict batch` begins.
        ...
        
    def on_predict_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'step_outputs', batch: Any, batch_index: int, dataloader_index: Optional[int] = 0) -> None:
        #@: Called when the `predict batch` ends. 
        ...
    
    
    def on_train_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the train begins. 
        ...
    
    
    def on_train_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the train ends. 
        ...
        
    
    def on_validation_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the validation loop begins. 
        ...
    
    
    def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the validation loop ends. 
        ...
       
    
    def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the test begins. 
        ...
    
    
    def on_test_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the test ends. 
        ... 
        
    
    def on_predict_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the predict begins. 
        ...
    
    
    def on_predict_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called when the predict ends. 
        ... 
    
    
    def on_exception(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', exception: BaseException) -> None:
        #@: Called when any trainer excution is interrupted by an exception.
        ...
        
        
    def state_dict(self) -> dict[str, Any]:
        #@: Called when saving a checkpoint, implement to generate callback's `state_dict`
        # Returns:
        #   A dictionary containing callback state.
        return {}
    
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        #@: Called when loading a checkpoint, implement to reload callback state given callback's `state_dict`
        # Args:
        #   state_dict: the callback state returned bt `state_dict`
        ...
        
        
    def on_save_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
        #@: Called when saving a checkpoint to give you a chance to store anything else you might want to save.
        ...
        
    
    def on_load_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
        #@: Called when loading a model checkpoint, use to reload state. 
        ...
        
    
    def on_before_backward(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', loss: torch.tensor) -> None:
        #@: Called before `loss.backward()`
        ...
        
    def on_after_backward(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        #@: Called after `loss.backward()` and before `optimizers are stepped`
        ...
        
    def on_before_optimizer_step(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', optimizer: torch.optim.Optimizer) -> None:     
        #@: Called before `optimizer.step()`
        ...
        
    def on_before_zero_grad(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', optimizer: torch.optim.Optimizer) -> None:
        #@: Called before `optimizer.zero_grad()`
        ...
        
