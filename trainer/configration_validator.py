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
#@: NOTE : ALTER LATER 
import sys
sys.path.append("/Users/rahulsawhney/Library/CloudStorage/OneDrive-Personal/FlameModules")
#@: NOTE : ALTER LATER 
from lightning.fabric.utilities.warnings import PossibleUserWarning
from states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
#from lightning.pytorch.utilities.imports import _graphcore_available_and_importable :: NOTE :: ERROR
#from utilities.imports import _graphcore_available_and_importable :: NOTE :: ERROR 
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature


# Function _verify_loop_configurations:
# This function validates the model's and trainer's configurations based on the current training state. 
# It ensures that the model and trainer are set up correctly before the main training, validating, 
# testing, or predicting loops are executed.
#
# Parameters:
#       -  trainer: An instance of 'flame_modules.Trainer'. This handles the training, validation, testing, and predicting processes.
#
# Local Variables:
#       -  model: An instance of 'FlameModule', extracted from the trainer. It's the main model to be trained or evaluated.
#
# Procedure:
#       1. Check if the 'state.fn' attribute of the trainer is set. If not, raise a ValueError.
#       2. Depending on the value of 'state.fn', perform the appropriate validation checks:
#              a) TrainerFn.Fitting: Ensure the training and validation loops are configured correctly. Additionally, verify the manual optimization configuration.
#              b) TrainerFn.Validating: Validate the setup for the validation loop.
#              c) TrainerFn.Testing: Validate the setup for the testing loop.
#              d) Trainer.Predicting: Validate the configuration for the prediction loop.
#       3. Independently of 'state.fn', verify the batch transfer support and model configuration.
#       4. Raise a warning if there are issues with the data loader iteration.
#
# Design Considerations:
#       1. Comprehensive Validation: The function ensures that, regardless of the training state, all necessary configurations are validated, preventing potential run-time issues.
# 
#       2. Clear Code Structure: The code is organized in a way that each training state has its own dedicated block, making the function easy to read and understand.
# 
#       3. Modular Validation: Instead of incorporating all the validation logic within this function, it delegates specific validation tasks to dedicated helper functions, promoting code reusability and clarity.
#
# Key Takeaways:
# The `_verify_loop_configurations` function acts as a central validation hub that ensures the trainer and model configurations are set up correctly. By thoroughly validating all aspects of the trainer and model setup based on the current training state, this function plays a crucial role in ensuring a smooth and error-free training and evaluation experience for users of the "FlameModules" framework.

def _verify_loop_configurations(trainer: 'flame_modules.Trainer') -> None:
    #@: Checks that the model is configured correctly before the run is started
    model: 'FlameModule' = trainer.flame_module

    if trainer.state.fn is None:
        raise ValueError('Trainer State fn must be set before validating loop configuration')
    if trainer.state.fn == TrainerFn.Fitting:
        __verify_train_val_loop_configuration(trainer, model)
        __verify_manual_optimization_support(trainer, model)
    
    elif trainer.state.fn == TrainerFn.Validating:
        __verify_eval_loop_configuration(model, 'val')
    
    elif trainer.state.fn == TrainerFn.Testing:
        __verify_eval_loop_configuration(model, 'test')
    
    elif trainer.state.fn == Trainer.Predicting:
        __verify_eval_loss_configuration(mode, 'predict')
        
    __verify_batch_transfer_support(trainer)
    
    __verify_configure_model_configuration(model)
    
    __warn_dataloader_iter_limitations(model)
    


# Function __verify_train_val_loop_configuration:
# This function is designed to validate the configuration of the training and validation loops within 
# the "FlameModules" framework. It ensures that the required methods for training and validation, 
# such as 'training_step', 'configure_optimizers', etc., are defined in the model before training begins.
#
# Parameters:
#       -  trainer: An instance of 'flame_modules.Trainer'. It's responsible for the entire training and validation process.
#       -  model: An instance of 'flame_modules.FlameModule'. The model being trained or evaluated.
#
# Local Variables:
#       -  has_training_step: A boolean that checks if 'training_step' is overridden in the model.
#       -  has_optimizers: A boolean to verify if 'configure_optimizers' is defined in the model.
#       -  has_val_loader: A boolean that checks if a validation data loader is defined in the trainer's validation loop.
#       -  has_val_step: A boolean that checks if 'validation_step' is overridden in the model.
#
# Procedure:
#       1. Check if 'training_step' is defined. If not, raise a MisconfigurationException.
#       2. Verify that 'configure_optimizers' is defined in the model. If missing, raise a MisconfigurationException.
#       3. Determine if a validation data loader and 'validation_step' are provided. Based on their presence or absence, 
#          appropriate warnings are issued:
#              a) If 'val_dataloader' is provided but 'validation_step' is missing, a warning is raised about skipping the validation loop.
#              b) If 'validation_step' is defined but there's no 'val_dataloader', a warning is issued indicating that the validation loop will be skipped.
#
# Design Philosophy:
#       1. Defensive Programming: By checking for the presence of crucial methods before the training or validation 
#          process begins, the function ensures a smoother user experience and prevents potential run-time errors.
# 
#       2. Informative Feedback: The function provides clear error messages and warnings, guiding the user about 
#          missing components or potential configuration issues.
#
#       3. Modular Design: The function encapsulates a specific validation logic related to the training and validation loops, 
#          making the overall codebase organized and maintainable.
#
# By ensuring that all required components are in place before the training and validation process starts, the 
# `__verify_train_val_loop_configuration` function significantly improves the reliability and user-friendliness of the "FlameModules" framework.
    
def __verify_train_val_loop_configuration(trainer: 'flame_modules.Trainer', model: 'flame_modules.FlameModule') -> None:
    has_training_step = is_overridden('training_step', model)
    if not has_training_step:
        raise MisconfigurationException(
            'No `training_step()` method defined. FlameModules `Trainer` excepts `training_step()`, `train_dataloader()` and `configure_optimizers()` methods to be defined.'
        )    

    has_optimizers: is_overridden('configure_optimizers', model) 
    if not has_optimizers:
        raise MisconfigurationException(
            'No `configure_optimizers()` method defined. FlameModules `Trainer` excepts `training_step()`, `train_dataloader()` and `configure_optimizers()` methods to be defined.'
        )

    has_val_loader = trainer.fit_loop.epoch_loop.val_loop._data_source.is_defined()
    has_val_step = is_overridden('validation_step', model)
    
    if has_val_loader and not has_val_step:
        rank_zero_warn('You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.')
    
    if has_val_step and not has_val_loader:
        rank_zero_warn(
            'You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.',
            category= PossibleUserWarning,
        )
    

# Function __verify_eval_loop_configuration:
# This function serves to validate the configuration of evaluation loops within the "FlameModules" ecosystem. 
# Depending on the provided 'stage' parameter, the function checks if the corresponding methods like 'validation_step', 
# 'predict_step' etc., are properly defined in the model for a smooth evaluation process.
#
# Parameters:
#       -  model: An instance of 'flame_modules.FlameModule'. The model being trained or evaluated.
#       -  stage: A string representing the current evaluation stage, e.g., 'val', 'predict', etc.
#
# Local Variables:
#       -  step_name: The method name in the model that corresponds to the provided stage.
#       -  has_step: A boolean flag indicating if the method (like 'validation_step', 'predict_step') is overridden in the model.
#
# Procedure:
#       1. The function constructs the step_name based on the provided stage.
#       2. It then checks if the model has this method overridden.
#       3. If the stage is 'predict', additional checks are conducted:
#               a) It verifies that 'predict_step' is defined in the model.
#               b) If 'predict_step' isn't overridden, it checks if the 'forward' method is defined.
#       4. For stages other than 'predict', it checks if the corresponding method is overridden in the model.
#       5. If any of these checks fail, a MisconfigurationException is raised, informing the user about the missing methods.
#
# Design Philosophy:
#       1. Dynamic Method Checking: The function dynamically constructs method names based on the provided stage, ensuring
#                                   flexibility and reducing redundancy in the codebase.
# 
#       2. User Feedback: By raising a MisconfigurationException with a descriptive message, the function offers immediate 
#                          feedback to the user about what's missing or misconfigured.
#
#       3. Modular Approach: This function encapsulates a specific validation logic related to evaluation loops, which helps 
#                             in maintaining a clean and organized codebase.
#
# The `__verify_eval_loop_configuration` function is integral to ensuring the robustness of the "FlameModules" ecosystem. By 
# making sure that users define necessary methods for evaluation, it helps in preventing potential run-time errors, thus enhancing 
# user experience and model evaluation accuracy.

def __verify_eval_loop_configuration(model: 'flame_modules.FlameModule', stage: str) -> None:
    step_name: str = 'validation_step' if stage == 'val' else f'{stage}_step'
    has_step = is_overridden(step_name, model)
    
    if stage == 'predict':
        if model.predict_step is None:
            raise MisconfigurationException('`predict_step` cannot be None to run `Trainer.predict`')
        if not has_step and not is_overridden('forward', mode):
            raise MisconfigurationException('`Trainer.predict` requires `forward` method to run.')
        
    else:
        if not has_step:
            trainer_method: str = 'validate' if stage == 'val' else stage
            raise MisconfigurationException(
                f'No `{step_name}()` method defined to run `Trainer.{trainer_method}`.'
            )      
            

# Function __verify_batch_transfer_support:
# Within the "FlameModules" ecosystem, this function's primary role is to ensure the compatibility of specific batch transfer 
# hooks with the IPU (Intelligence Processing Unit) accelerator. It aims to prevent unintended and unsupported operations 
# when using IPUs by validating that specific batch transfer hooks haven't been overridden.
#
# Parameters:
#       -  trainer: An instance of 'flame_modules.Trainer'. The trainer instance which contains information about the chosen 
#                   accelerator and other training configurations.
#
# Local Variables:
#       -  batch_transfer_hooks: A tuple of strings representing names of the hooks responsible for batch transfer operations.
#       -  datahook_selector: An object extracted from the trainer, aiding in the selection and execution of data-related hooks.
#
# Procedure:
#       1. The function first ensures that the datahook_selector is initialized and not None.
#       2. It then iterates through each hook in the batch_transfer_hooks.
#       3. For each hook, the function checks if the Graphcore library (which supports IPUs) is available and importable.
#       4. If the current trainer is using the IPUAccelerator and the batch transfer hook is overridden either in the model or 
#          the data module, a MisconfigurationException is raised. This informs the user that the overridden operation is not 
#          compatible with IPUs.
#
# Design Philosophy:
#       1. Preventative Error Handling: By preemptively checking for incompatible configurations, the function acts as a 
#                                       protective layer, ensuring that the user is immediately informed about potential 
#                                       pitfalls when using IPUs.
# 
#       2. Modular & Clean Code: The function encapsulates a specific validation logic, making the codebase modular and easy 
#                                to maintain.
#
#       3. Informative Feedback: By raising a clear MisconfigurationException, the function provides the user with direct feedback 
#                                about the issue and what needs to be corrected.
#
# The `__verify_batch_transfer_support` function showcases "FlameModules" dedication to ensuring a smooth user experience by 
# preventing configuration errors. By explicitly guiding users away from unsupported operations, it helps streamline the 
# development process when using advanced accelerators like IPUs.

def __verify_batch_transfer_support(trainer: 'flame_modules.Trainer') -> None:
    batch_transfer_hooks: tuple[str] = ('transfer_batch_to_device', 'on_after_batch_transfer')
    datahook_selector = trainer._data_connector._datahook_selector
    assert datahook_selector is not None
    
    for hook in batch_transfer_hooks:
        if _graphcore_available_and_importable():
            from lightning_graphcore import IPUAccelerator

            if isinstance(trainer.accelerator, IPUAccelerator) and (
                is_overridden(hook, datahook_selector.model) or is_overridden(hook, datahook_selector.datamodule)
            ):
                raise MisconfigurationException(f'Overriding `{hook}` is not supported with IPUs.')



# Function __verify_manual_optimization_support:
# Within the "FlameModules" framework, this function serves to validate the compatibility of manual optimization settings 
# with certain trainer configurations. It ensures that features exclusive to automatic optimization aren't mistakenly 
# activated when manual optimization is in use.
#
# Parameters:
#       -  trainer: An instance of 'flame_modules.Trainer'. The trainer instance containing optimization settings and configurations.
#       -  model: An instance of a class derived from 'flame_modules.FlameModule'. The model encapsulating the optimization mode 
#                 (automatic or manual).
#
# Procedure:
#       1. If the model uses automatic optimization, the function returns immediately, requiring no further checks.
#       2. If gradient clipping is enabled in the trainer and the model uses manual optimization, a MisconfigurationException 
#          is raised, alerting the user to the incompatibility.
#       3. Similarly, if gradient accumulation is not set to its default value and manual optimization is in use, another 
#          MisconfigurationException is raised to notify the user.
#
# Design Philosophy:
#       1. Robustness & Reliability: The function acts as a safety net, preventing users from inadvertently mixing configurations 
#                                    that are incompatible, ensuring robust and predictable behavior.
# 
#       2. Informative Error Handling: By raising specific exceptions with detailed messages, the function guides the user towards 
#                                      rectifying the configuration incompatibilities, offering a more user-friendly experience.
#
#       3. Simplification & Clarification: By enforcing these checks, "FlameModules" simplifies the optimization process, eliminating 
#                                          potential sources of confusion and ensuring clarity in how optimization settings should be used.
#
# The `__verify_manual_optimization_support` function exemplifies "FlameModules" commitment to creating an intuitive and error-resistant 
# development experience. It acts as a guardian, ensuring that the user's configuration aligns with the best practices and intended 
# functionalities of the framework.

def __verify_manual_optimization_support(trainer: 'flame_modules.Trainer', model: 'flame_modules.FlameModule') -> None:
    if model.automatic_optimization:
        return 
    
    if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
        raise MisconfigurationException(
            'Automatic gradient clipping is not supported for manual optimization. Switch to Automatic Optimization'
        )
        
    if trainer.accumulate_grad_batches != 1:
        raise MisconfigurationException(
            'Automatic gradient accumulation is not supported for manual optimization. Switch to Automatic Optimization'
        )


# Function __verify_dataloader_iter_limitations:
# Within the "FlameModules" ecosystem, this function verifies potential pitfalls when using the `dataloader_iter` step flavor 
# in the provided model. It checks for any unintended consequences that might arise due to the multiple consumption of the iterator 
# within a single step.
#
# Parameters:
#       -  model: An instance of a class derived from 'flame_modules.FlameModule'. The model being checked for the use of 
#                 `dataloader_iter` in its step functions.
#
# Procedure:
#       1. The function iterates through all the step functions in the model: `training_step`, `validation_step`, `predict_step`, 
#          and `test_step`.
#       2. For each step function that is not None, it checks if the signature of the function has 'dataloader_iter' as a parameter.
#       3. If any of the step functions use `dataloader_iter`, a warning is issued to the user detailing the potential complications.
#
# Design Philosophy:
#       1. Proactive Error Prevention: By providing warnings for experimental or potentially problematic features, the function aims 
#                                      to preemptively prevent misconfigurations or misunderstandings.
# 
#       2. User-Centric Feedback: The function gives explicit feedback, explaining the implications of using `dataloader_iter` and 
#                                 guiding users to be cautious.
#
#       3. Future-Proofing: By flagging experimental features, "FlameModules" emphasizes the potential for change, ensuring that developers 
#                           are aware of the evolving nature of such features.
#
#       4. Iterative Improvement: Warning users about experimental features provides an avenue for gathering feedback, which can be 
#                                 crucial for refining and improving the feature in subsequent releases.
#
# The `__verify_dataloader_iter_limitations` function stands as a testament to "FlameModules" commitment to user-friendly interfaces and 
# responsible feature development. By providing explicit guidance on experimental features, it ensures that developers can use them 
# with a full understanding of their nuances and limitations.

def __verify_dataloader_iter_limitations(model: 'flame_modules.FlameModule') -> None:
    #@: Check if `dataloader_iter is enabled`
    if any(
        is_param_in_hook_signature(step_fn, 'dataloader_iter', explicit= True)
        for step_fn in (model.training_step, model.validation_step, model.predict_step, model.test_step) 
        if step_fn is not None 
    ):
        rank_zero_warn(
            "You are using the `dataloader_iter` step flavor. If you consume the iterator more than once per step, the"
            " `batch_idx` argument in any hook that takes it will not match with the batch index of the last batch"
            " consumed. This might have unforeseen effects on callbacks or code that expects to get the correct index."
            " This will also no work well with gradient accumulation. This feature is very experimental and subject to"
            " change.",
            category= PossibleUserWarning
        )


# Function __verify_configure_model_configuration:
# Within the "FlameModules" ecosystem, this function verifies if the correct model configuration methods are being used. It checks 
# for deprecated or conflicting method overrides in the provided model, ensuring compatibility with newer module standards.
#
# Parameters:
#       -  model: An instance of a class derived from 'flame_modules.FlameModule'. This is the model being checked for method overrides.
#
# Raises:
#       -  RuntimeError: If both 'configure_sharded_model' and 'configure_model' are overridden in the provided model, signaling 
#                       conflicting configurations.
#
# Procedure:
#       1. The function checks if 'configure_sharded_model' is overridden in the provided model.
#       2. If it is, it then checks if 'configure_model' is also overridden.
#       3. If both methods are overridden, a RuntimeError is raised indicating the conflict and suggesting the appropriate action.
#       4. If only 'configure_sharded_model' is overridden, a deprecation warning is logged, advising the user to switch to the newer 
#          'configure_model' method.
#
# Design Philosophy:
#       1. Compatibility Assurance: By checking for deprecated method overrides, the function ensures that developers are using 
#                                   the latest, more efficient configurations.
# 
#       2. User-Centric Feedback: In cases of misconfiguration, the function provides clear and actionable feedback, guiding users 
#                                 towards the right approach.
#
#       3. Future-Proofing: With this verification step, "FlameModules" is setting the stage for future improvements, ensuring that 
#                           legacy configurations do not hinder the adoption of new features or enhancements.
#
#       4. Centralized Verification: Instead of relying on scattered checks throughout the codebase, this function offers a centralized 
#                                    verification step, enhancing maintainability and clarity.
#
# The `__verify_configure_model_configuration` function exemplifies "FlameModules" commitment to providing a smooth and error-free 
# experience for developers. By guiding them away from deprecated methods and towards updated configurations, it ensures optimal 
# performance and compatibility.

def __verify_configure_model_configuration(model: 'flame_modules.FlameModule') -> None:
    if is_overridden('configure_sharded_model', model):
        name: str = type(model).__name__
        if is_overridden('configure_model', model):
            raise RuntimeError(
                f'Both `{name}.configure_model`, and `{name}.configure_sharded_model` are overridden. The latter is' 
                f'deprecated and it should be replaced with the former.'
            )

        rank_zero_deprecation(
            f'You have overridden `{name}.configure_sharded_model` which is deprecated. Please override the'
            f'`configure_model` hook instead. Instantiation with the newer hook will be created on the device right'
            f'away and have the right data type depending on the precision setting in the Trainer.'
        )

