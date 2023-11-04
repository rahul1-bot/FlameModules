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
import logging
from copy import deepcopy
from FlameModules.trainer.states import TrainerStatus
from FlameModules.callbacks.checkpoint import Checkpoint
from FlameModules.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.exceptions import _TunerExitException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin

log = logging.getLogger(__name__)


def _call_and_handle_interrupt(trainer: 'flame_modules.Trainer', trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    #@: Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    # as all errors should funnel through them. 
    #
    #   Args:
    #       - trainer_fn: One of `fit`, `validate`, `test`, `predict`
    #       - *args: positional arguments to be passed to the `trainer_fn`
    #       - **kwargs: keyword arguments to be passed to `trainer_fn`
    try:
        if trainer.strategy.launcher is not None:
            return trainer.strategy.launcher.launch(trainer_fn, *args, trainer= trainer, **kwargs)
        return trainer_fn(*args, **kwargs)
    
    except _TunerExitException:
        _call_teardown_hook(trainer)
        trainer._teardown()
        trainer.state.status = TrainerStatus.Finished
        trainer.state.stage = None
        
        # :: NOTE :: ToDo Unify Both exceptions, where `KeyboardError` doesn't re-raise.
        
        
        


def _call_setup_hook(trainer: 'flame_modules.Trainer') -> None:
    # Purpose:
    #     The `_call_setup_hook` function is intended to initialize and set up the training environment within the 
    #     FlameModules framework. It ensures that all necessary components such as modules, loggers, and strategies are 
    #     configured and ready before the training starts.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, which is the central orchestrating entity in the FlameModules 
    #                training lifecycle.
    #
    # Key Steps:
    #   1. Verify that the `trainer`'s current state function (`state.fn`) is set, indicating the training stage (e.g., 
    #      training, validating, testing, etc.).
    #   
    #   2. Iterate through all modules in `trainer.flame_module` and assign the `trainer.strategy.root_device` to the 
    #      module's device attribute if the module inherits from `_DeviceDtypeModuleMixin`. This ensures that all modules 
    #      are on the correct device before the setup.
    #
    #   3. Loop through all loggers associated with the `trainer` and initialize their `experiment` attribute, if it exists. 
    #      This could involve setting up connections to logging backends or similar preparatory steps.
    #
    #   4. Call the `trainer.strategy.barrier` with 'pre_setup', which acts as a synchronization point across multiple devices 
    #      or nodes before setup begins.
    #
    #   5. If a `datamodule` is present, invoke the `setup` hook of the `datamodule` with the current stage. This typically 
    #      involves preparing data loaders and applying transformations specific to the stage.
    #   
    #   6. Invoke the `setup` hook for all registered callbacks and the main `flame_module`, which might include steps like 
    #      resetting states or applying initial configurations.
    #
    #   7. Finally, call the `trainer.strategy.barrier` with 'post_setup', ensuring that all setup processes are completed 
    #      across the board before moving forward.

    assert trainer.state.fn is not None
    fn = trainer.state.fn
    
    for module in trainer.flame_module.modules():
        if isinstance(module, _DeviceDtypeModuleMixin):
            module._device = trainer.strategy.root_device
        
    for logger in trainer.loggers:
        if hasattr(logger, 'experiment'):
            _ = logger.experiment
            
    trainer.strategy.barrier('pre_setup')
    
    if trainer.datamodule is not None:
        _call_flame_datamodule_hook(trainer, 'setup', stage= fn)
    _call_callback_hook(trainer, 'setup', stage= fn)
    _call_flame_module_hook(trainer, 'setup', stage= fn)
    
    trainer.strategy.barrier('post_setup')
    
    



def _call_teardown_hook(trainer: 'flame_modules.Trainer') -> None:
    # Purpose:
    #     The `_call_teardown_hook` function serves to gracefully dismantle the training environment within the FlameModules 
    #     framework after the completion of a training stage. It systematically releases resources, finalizes loggers, and 
    #     performs cleanup operations to ensure that the system is in a stable state post-training.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, which plays the central role in managing the training process 
    #                within the FlameModules ecosystem.
    #
    # Key Steps:
    #   1. Validate that the `trainer`'s current state function (`state.fn`) is set, confirming that the stage of training 
    #      (e.g., training, validating, testing) to be torn down is known.
    #   
    #   2. If a `datamodule` is associated with the `trainer`, invoke its `teardown` method with the current stage passed 
    #      as an argument. This is intended to clean up data loaders or any related data structures.
    #
    #   3. Trigger the `teardown` hooks for all callbacks associated with the `trainer`. This allows custom cleanup logic 
    #      in callbacks to execute, such as closing files or releasing network connections.
    #
    #   4. Call the `teardown` method on the main `flame_module`. This step typically involves finalizing model states, 
    #      flushing buffers, or other post-training cleanup actions.
    #
    #   5. Reset the `flame_module`'s current function name and metric attributes to `None`. This indicates that no 
    #      particular training function is currently being executed and that metric tracking is reset.
    #
    #   6. Loop through each logger in `trainer.loggers` and call the `finalize` method with a 'success' argument, signaling 
    #      that the training stage has ended successfully. This step often includes actions like closing log files or writing 
    #      final entries to a database.
    #
    #   7. Call the `describe` method on the `trainer.profiler`, which typically outputs a summary of profiling information 
    #      such as execution times and resource usage statistics. This can be valuable for analyzing and optimizing the training process.

    assert trainer.state.fn is not None
    fn = trainer.state.fn
    
    if trainer.datamodule is not None:
        _call_flame_datamodule_hook(trainer, 'teardown', stage= fn)
    
    _call_callback_hooks(trainer, 'teardown', stage= fn)
    _call_flame_module_hook(trainer, 'teardown', stage= fn)
    
    trainer.flame_module._current_fx_name = None
    trainer.flame_module._metric_attributes = None
    
    for logger in trainer.loggers:
        logger.finalize('success')
        
    trainer.profiler.describe()
    
    
    
    
    
    
def _call_flame_module_hook(trainer: 'flame_modules.Trainer', hook_name: str, *args: Any, flame_module: 'flame_modules.FlameModule', **kwargs: Any) -> Any:
    # Purpose:
    #     The `_call_flame_module_hook` function is a centralized mechanism for invoking specific hooks on the `FlameModule` 
    #     during different stages of the training process. It ensures that any custom behavior defined in these hooks is 
    #     executed at the appropriate time, enabling customization of the training loop.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer` that manages the overall training process.
    #   - `hook_name`: A string representing the name of the hook to be called on the `FlameModule`.
    #   - `*args`: Variable length argument list that will be passed to the hook.
    #   - `flame_module`: An optional instance of `flame_modules.FlameModule` on which the hook will be called. If not 
    #                     provided, the default `flame_module` associated with the `trainer` will be used.
    #   - `**kwargs`: Arbitrary keyword arguments that will be passed to the hook.
    #
    # Key Steps:
    #   1. Log the invocation of the hook for debugging purposes.
    #   
    #   2. Determine the `FlameModule` instance to be used by either using the one provided as a parameter or the one 
    #      associated with the `trainer`.
    #
    #   3. Check if the `FlameModule` instance is present; raise a `TypeError` if not, since hooks cannot be called without 
    #      a `FlameModule`.
    #
    #   4. Retrieve the function (`fn`) corresponding to the `hook_name` from the `FlameModule` and verify that it is callable. 
    #      If it's not, return `None` as hooks that don't exist cannot be called.
    #
    #   5. Store the current function name (`_current_fx_name`) to restore it later, and set the `hook_name` as the new 
    #      current function name.
    #
    #   6. Profile the hook execution with the `trainer`'s profiler to measure performance and resource usage, providing 
    #      insights into the efficiency of the hook.
    #
    #   7. Execute the hook by calling the retrieved function `fn` with the provided arguments (`*args` and `**kwargs`), 
    #      capturing any output that the function may return.
    #
    #   8. Restore the previous function name (`_current_fx_name`) after the hook execution to maintain the state of the 
    #      `FlameModule`.
    #
    #   9. Return any output from the hook function, which could be used by other parts of the `trainer` for further processing.

    log.debug(f'{trainer.__class__.__name__}: calling flame module hook: {hook_name}')
    flame_module = flame_module or trainer.flame_module
    
    if flame_module is None:
        raise TypeError('No `FlameModule` is available to call hooks on')
    
    fn = getattr(flame_module, hook_name)
    if not callable(fn):
        return None
    
    prev_fx_name = flame_module._current_fx_name
    flame_module._current_fx_name = hook_name
    
    with trainer.profiler.profile(f'[FlameModule]{flame_module.__class__.__name__}.{hook_name}'):
        output = fn(*args, **kwargs)
        
    flame_module._current_fx_name = prev_fx_name
    return output     
        





def _call_flame_datamodule_hook(trainer: 'flame_modules.Trainer', hook_name: str, *args: Any, **kwargs: Any) -> None:
    # Purpose:
    #     The `_call_flame_datamodule_hook` function's role is to trigger specific hooks on the `FlameDataModule` at certain 
    #     points in the training lifecycle. This enables the `FlameDataModule` to execute code that prepares data, cleans up 
    #     resources, or performs other data-related tasks at the right time.
    #
    # Parameters:
    #   - `trainer`: The `flame_modules.Trainer` instance managing the training process.
    #   - `hook_name`: The name of the hook to invoke on the `FlameDataModule`.
    #   - `*args`: Additional positional arguments passed to the hook function.
    #   - `**kwargs`: Additional keyword arguments passed to the hook function.
    #
    # Key Steps:
    #   1. Log the hook invocation for debugging and transparency.
    #
    #   2. Ensure the presence of a `FlameDataModule` within the `trainer`. If it is absent, a `TypeError` is raised 
    #      indicating that hooks cannot be called.
    #
    #   3. Retrieve the hook function (`fn`) from the `FlameDataModule` using the `hook_name` provided.
    #
    #   4. If the retrieved function is callable, proceed to the next step. If not, the function ends and returns `None`, 
    #      indicating no operation was performed.
    #
    #   5. Profile the execution of the hook using the `trainer`'s profiler to gather performance data and insights, which 
    #      can be important for optimization and debugging.
    #
    #   6. Execute the hook by calling the function `fn` with any additional arguments provided, capturing any output or 
    #      return value.
    #
    #   7. If the hook execution is successful and a value is returned, this value is returned to the caller. If there is 
    #      no return value or if the hook is not callable, return `None`.
    #
    #   8. This function is integral to the extensibility and flexibility of the FlameModules framework, allowing data-related 
    #      behaviors to be dynamically included in the training workflow.

    log.debug(f'{trainer.__class__.__name__}: calling flame datamodule hook: {hook_name}')
    
    if trainer.datamodule is None:
        raise TypeError('No `FlameDataModule` is available to call hooks on')
    
    fn = getattr(trainer.datamodule, hook_name)
    if callable(fn):
        with trainer.profiler.profile(f'[FlameDataModule]{trainer.datamodule.__class__.__name__}.{hook_name}'):
            return fn(*args, **kwargs)
        
    return None




def _call_callback_hooks(trainer: 'flame_modules.Trainer', hook_name: str, *args: Any, monitoring_callbacks: Optional[bool] = None, **kwargs: Any) -> None:
    # Purpose:
    #     The `_call_callback_hooks` function is responsible for invoking specific hooks on the callbacks registered with 
    #     the trainer. It provides a mechanism to trigger custom callback logic at predetermined points in the training 
    #     lifecycle, such as at the start/end of training, on batch or epoch completion, or when monitoring conditions are 
    #     met.
    #
    # Parameters:
    #   - `trainer`: The `flame_modules.Trainer` instance that coordinates the training process.
    #   - `hook_name`: The name of the hook to be invoked on each callback.
    #   - `*args`: Additional positional arguments to be passed to the hook.
    #   - `monitoring_callbacks`: Optional boolean to filter callbacks based on whether they are monitoring callbacks 
    #                             like `EarlyStopping` and `Checkpoint`.
    #   - `**kwargs`: Additional keyword arguments to be passed to the hook.
    #
    # Key Steps:
    #   1. Log the hook invocation for transparency and aid in debugging.
    #
    #   2. Temporarily update the `flame_module._current_fx_name` to the current hook's name, which helps in tracking the 
    #      execution flow.
    #
    #   3. Based on `monitoring_callbacks` parameter, filter the callbacks to either only monitoring-related ones or exclude 
    #      them, as per the caller's requirements.
    #
    #   4. Iterate over the applicable callbacks, and for each callback, retrieve the method corresponding to the `hook_name`.
    #
    #   5. If the retrieved method is callable, profile its execution for performance tracking and then invoke the hook 
    #      method with the provided arguments and keyword arguments.
    #
    #   6. After executing all hooks, restore the `flame_module._current_fx_name` to its previous value to maintain 
    #      accurate tracking of the FlameModule's state.
    #
    #   7. This function enhances the trainer's flexibility by allowing custom callbacks to participate in the training 
    #      process, thereby enabling sophisticated behavior like early stopping or model checkpointing.

    log.debug(f'{trainer.__class__.__name__}: calling callback hook: {hook_name}')
    flame_module = trainer.flame_module
    if flame_module:
        prev_fx_name = flame_module._current_fx_name
        flame_module._current_fx_name = hook_name
        
    callbacks = trainer.callbacks
    if monitoring_callbacks is True:
        callbacks = [
            cb for cb in callbacks if isinstance(cb, (EarlyStopping, Checkpoint))
        ]
    elif monitoring_callbacks is False:
        callbacks = [
            cb for cb in callbacks if not isinstance(cb, (EarlyStopping, Checkpoint))
        ]
        
    for callback in callbacks:
        fn = getattr(callback, hook_name)
        if callable(fn):
            with trainer.profiler.profile(f'[Callback]{callback.state_key}.{hook_name}'):
                fn(trainer, trainer.flame_module, *args, **kwargs)
                
    
    if flame_module:
        flame_module._current_fx_name = prev_fx_name
        
        
        



def _call_callbacks_state_dict(trainer: 'flame_module.Trainer') -> dict[str, dict[Any, Any]]:
    # Purpose:
    #     The `_call_callbacks_state_dict` function is designed to gather and return the state dictionaries of all callbacks 
    #     associated with the trainer. This is typically used for checkpointing, which allows for saving the state of each 
    #     callback so that training can be resumed from the same point if interrupted.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_module.Trainer`, which is the core orchestrator of training operations and 
    #                lifecycle events in the FlameModules framework.
    #
    # Returns:
    #   - A dictionary where each key is a callback's unique state key and the value is the corresponding state dictionary 
    #     of that callback.
    #
    # Key Steps:
    #   1. Initialize an empty dictionary to store the state dictionaries of callbacks.
    #
    #   2. Loop through each callback registered with the trainer.
    #
    #   3. Call the `state_dict` method of each callback to retrieve its state dictionary.
    #
    #   4. If a callback returns a non-empty state dictionary, store it in the `callback_state_dicts` using the callback's 
    #      unique `state_key` as the key.
    #
    #   5. After collecting all state dictionaries, return the `callback_state_dicts` dictionary.
    #
    #   6. This function ensures the persistence of critical callback states, enabling consistent behavior across training 
    #      sessions and robustness to interruptions.

    callback_state_dicts: dict[Any, Any] = {}
    for callback in trainer.callbacks:
        state_dict = callback.state_dict()
        if state_dict:
            callback_state_dicts[callback.state_key] = state_dict
    
    return callback_state_dicts






def _call_callbacks_on_save_checkpoint(trainer: 'flame_modules.Trainer', checkpoint: dict[str, Any]) -> None:
    # Purpose:
    #     The `_call_callbacks_on_save_checkpoint` function invokes the `on_save_checkpoint` hook for each callback 
    #     registered with the trainer. This hook is crucial for allowing each callback to contribute its state to the 
    #     checkpoint. This function ensures that all necessary states are saved so that the training session can be 
    #     resumed with the same configuration and history.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, which orchestrates the training process.
    #   - `checkpoint`: A dictionary that accumulates the state information of various components during checkpointing.
    #
    # Key Steps:
    #   1. Check if there is a `flame_module` present, and if so, preserve its current function name (fx_name).
    #
    #   2. Update the `flame_module`'s current function name to 'on_save_checkpoint', to reflect the operation being performed.
    #
    #   3. Iterate through all the callbacks associated with the trainer.
    #
    #   4. For each callback, profile the execution of its `on_save_checkpoint` method with the trainer's profiler to track performance.
    #
    #   5. Call the `on_save_checkpoint` method on each callback, allowing it to save its state into the `checkpoint` dictionary.
    #
    #   6. After all callbacks have been processed, if a `flame_module` was used, restore its previous function name.
    #
    #   7. This function ensures that the state of each callback is correctly captured in the checkpoint, facilitating 
    #      full restoration during training resumption.

    flame_module = trainer.flame_module
    if flame_module:
        prev_fx_name = flame_module._current_fx_name
        flame_module._current_fx_name = 'on_save_checkpoint'
        
    for callback in trainer.callbacks:
        with trainer.profiler.profile(f'[Callback]{callback.state_key}.on_save_checkpoint'):
            callback.on_save_checkpoint(trainer, trainer.flame_module, checkpoint)
            
    if flame_module:
        flame_module._current_fx_name = prev_fx_name
        
        




def _call_callbacks_on_load_checkpoint(trainer: 'flame_modules.Trainer', checkpoint: dict[str, Any]) -> None:
    # Purpose:
    #     The `_call_callbacks_on_load_checkpoint` function is responsible for the process of loading states from a 
    #     checkpoint into each registered callback during the resumption of training. This is essential for restoring the 
    #     training session to its previous state, including all the progress and configurations captured by the callbacks.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, which oversees the entire training process.
    #   - `checkpoint`: A dictionary containing the state information of various components that was previously saved during checkpointing.
    #
    # Key Steps:
    #   1. If a `flame_module` is present, store its current function name (fx_name).
    #
    #   2. Update the `flame_module`'s current function name to 'on_load_checkpoint' to indicate the restoration operation.
    #
    #   3. Retrieve the states of the callbacks from the `checkpoint` dictionary.
    #
    #   4. If the `callback_states` are not present in the checkpoint, exit the function as there is nothing to load.
    #
    #   5. Loop through all the callbacks associated with the trainer.
    #
    #   6. For each callback, profile the execution of its `on_load_checkpoint` method using the trainer's profiler for performance tracking.
    #
    #   7. Invoke the `on_load_checkpoint` method on each callback, allowing it to load its state from the `checkpoint` dictionary.
    #
    #   8. After all callbacks have loaded their states, if a `flame_module` was involved, revert to its previous function name.
    #
    #   9. This function ensures that the state of each callback is correctly retrieved from the checkpoint, enabling the training session 
    #      to continue from where it left off with consistency and accuracy.

    flame_module = trainer.flame_module
    if flame_module:
        prev_fx_name = flame_module._current_fx_name
        flame_module._current_fx_name = 'on_load_checkpoint'
        
    callback_states: Optional[dict[Union[Type, str], dict]] = checkpoint.get('callbacks')
    
    if callback_states is None:
        return 
    
    for callback in trainer.callbacks:
        with trainer.profiler.profile(f'[Callback]{callback.state_key}.on_load_checkpoint'):
            callback.on_load_checkpoint(trainer, trainer.flame_module, checkpoint)
            
    if flame_module:
        flame_module._current_fx_name = prev_fx_name
        
        
        
        
def _call_callbacks_load_state_dict(trainer: 'flame_modules.Trainer', checkpoint: dict[str, Any]) -> None:
    # Purpose:
    #     The `_call_callbacks_load_state_dict` function is designed to load the states of each callback from the 
    #     checkpoint dictionary into the current training session. This function is crucial when resuming training 
    #     from a checkpoint, as it ensures that the callbacks are restored to their previous state and can continue to 
    #     function as before the training was paused or interrupted.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, which is responsible for managing the training loop and all associated processes.
    #   - `checkpoint`: A dictionary containing the saved state information for the training session, including states specific to each callback.
    #
    # Key Steps:
    #   1. Extract the callback state dictionaries from the `checkpoint` under the 'callbacks' key.
    #
    #   2. If no callback states are found in the checkpoint, the function exits, as there are no states to restore.
    #
    #   3. Iterate over all callbacks registered to the trainer.
    #
    #   4. For each callback, retrieve its state using either its `state_key` or, if not found, its `_legacy_state_key` to 
    #      maintain compatibility with earlier versions.
    #
    #   5. If a state dictionary is retrieved, create a deep copy of it to prevent any unintentional modifications to the 
    #      checkpoint state.
    #
    #   6. Use the `load_state_dict` method of the callback to load its state from the copied dictionary.
    #
    #   7. By the end of this function, all callbacks should have their states loaded from the checkpoint, allowing them to
    #      seamlessly continue their roles in monitoring, saving, or altering the training process.

    callback_states: Optional[dict[Union[Type, str], dict]] = checkpoint.get('callbacks')
    if callback_states is None:
        return 
    
    for callback in trainer.callbacks:
        state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
        if state:
            state = deepcopy(state)
            callback.load_state_dict(state)
            



def _call_strategy_hook(trainer: 'flame_modules.Trainer', hook_name: str, *args: Any, **kwargs: Any) -> Any:
    # Purpose:
    #     The `_call_strategy_hook` function is a core part of the FlameModules framework that is designed to invoke 
    #     the corresponding method (hook) on the trainer's strategy object. The strategy object dictates how the training
    #     process is distributed and managed across different hardware configurations, such as single GPUs, multiple GPUs,
    #     TPUs, or even across multiple nodes in a cluster. By calling strategy-specific hooks, the function allows the
    #     training strategy to participate in different stages of the training lifecycle and execute strategy-specific logic.
    #
    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, the orchestrator of the training process.
    #   - `hook_name`: A string representing the name of the method (hook) to be called on the strategy object.
    #   - `*args`, `**kwargs`: Additional arguments and keyword arguments to be passed to the hook method.
    #
    # Key Steps:
    #   1. Log a debug message indicating that a strategy hook is being called.
    #
    #   2. Retrieve the current function name being executed within the flame module to be able to restore it later.
    #
    #   3. Set the flame module's current function name to the hook that is being called to provide context for any 
    #      operations within the hook.
    #
    #   4. Obtain a reference to the method corresponding to `hook_name` from the trainer's strategy object.
    #
    #   5. Check if the obtained reference is callable. If it is not, the function returns `None`, indicating that the 
    #      hook does not exist or is not executable.
    #
    #   6. Profile the execution of the hook using the trainer's profiler to monitor performance and resource usage.
    #
    #   7. Call the hook method with the provided arguments and keyword arguments, capturing any output it may produce.
    #
    #   8. Restore the flame module's current function name to its previous value, maintaining the internal state consistency.
    #
    #   9. Return the output of the hook method, if any, allowing the caller to use it for further processing or decision-making.

    log.debug(f'{trainer.__class__.__name__}: calling strategy hook: {hook_name}')
    
    flame_module = trainer.flame_module
    prev_fx_name = flame_module._current_fx_name
    flame_module._current_fx_name = hook_name
    
    fn = getattr(trainer.strategy, hook_name)
    if not callable(fn):
        return None
    
    with trainer.profiler.profile(f'[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}'):
        output = fn(*args, **kwargs)
        
    flame_module._current_fx_name = prev_fx_name
    return output
        

