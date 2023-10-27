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
from dataclasses import dataclass
from lightning.pytorch.utilities.enums import LightningEnum

# Class TrainerStatus:
# The `TrainerStatus` is an enumeration within the "FlameModules" ecosystem, designed to succinctly represent the various operational 
# states that the `flame_modules.trainer.trainer.Trainer` can be in during its lifecycle.
#
# Values:
#       -  Initializing: Represents the initial phase when the trainer is being set up.
#       -  Running: Signifies that the trainer is actively executing its training process.
#       -  Finished: Denotes that the trainer has successfully completed its task.
#       -  Interrupted: Indicates that the trainer's execution was halted prematurely due to external factors.
#
# Property:
#       -  stopped: A boolean property that checks if the trainer's status is either `Finished` or `Interrupted`, thus helping in 
#                   quick assessments of whether the training process has halted.
#
# Design Philosophy:
#       1. Clarity & Precision: By encapsulating distinct trainer states into explicit enumeration values, `TrainerStatus` ensures 
#                                that the current state of the trainer is clearly and precisely communicated.
# 
#       2. Rapid State Assessment: The `stopped` property offers a quick mechanism to assess if the trainer's execution has halted, 
#                                  eliminating the need for manual state comparisons.
#
#       3. Modularity & Expandability: With an enumeration-based design, introducing new states or modifying existing ones becomes 
#                                      effortless, ensuring that the class remains adaptable to future changes.
#
#       4. Safety & Consistency: Leveraging the power of enumerations ensures type safety and consistency, reducing potential 
#                                pitfalls associated with string-based status checks.
#
# The `TrainerStatus` enum acts as a cornerstone in the "FlameModules" training ecosystem, establishing a standardized, clear, and 
# robust mechanism for representing and querying the operational states of the trainer.

class TrainerStatus(LightningEnum):
    #@: Enum for the `Status` of the `flame_modules.trainer.trainer.Trainer`
    Initializing: str = 'initializing'
    Running: str = 'running'
    Finished: str = 'finished'
    Interrupted: str = 'interrupted'
    
    @property
    def stopped(self) -> bool:
        return self in (self.Finished, self.Interrupted)
    

# Class TrainerFn:
# The `TrainerFn` serves as a comprehensive enumeration within the "FlameModules" framework, precisely capturing the various 
# user-facing functions that the `flame_modules.trainer.trainer.Trainer` can execute.
#
# Values:
#       -  Fitting: Corresponds to the `fit` function, indicating the trainer's primary training phase.
#       -  Validating: Refers to the validation phase typically accompanying the training process.
#       -  Testing: Represents the `test` function, suggesting that the trainer is in the evaluation phase post-training.
#       -  Predicting: Denotes the predictive phase where the trained model generates outputs for unseen data.
#
# Design Philosophy:
#       1. Explicit Function Mapping: The enumeration ensures that every core function of the trainer is mapped to a distinct value, 
#                                      making it easy to discern the current operation.
# 
#       2. Seamless Integration: Given its clear naming and structure, the `TrainerFn` can be effortlessly integrated with conditional 
#                                 logic or event-driven mechanisms within the training workflow.
#
#       3. Enhanced Code Readability: By employing an enumeration, the codebase achieves improved clarity and reduces potential 
#                                     ambiguities arising from string-based function checks.
#
#       4. Adaptability & Extendability: Should future versions of "FlameModules" introduce new functions or modify existing ones, 
#                                        the `TrainerFn` can be easily updated to reflect these changes.
#
# The `TrainerFn` enum stands as a pivotal element in the "FlameModules" training paradigm, offering a clear and standardized method 
# to represent and query the various operational functions of the trainer. Its design underscores the emphasis on clarity, modularity, 
# and user-centricity that "FlameModules" strives for.

class TrainerFn(LightningEnum):
    #@: Enum for the user-facing functions of the `flame_modules.trainer.trainer.Trainer` such as 
    #@: method: `flame_modules.trainer.trainer.Trainer.fit`
    #@: method: `flame_modules.trainer.trainer.Trainer.test`
    Fitting: str = 'fit'
    Validating: str = 'validate'
    Testing: str = 'test'
    Predicting: str = 'predict'
    

# Class RunningStage:
# Within the "FlameModules" framework, the `RunningStage` enum offers a granular representation of the trainer's current execution stage, 
# working harmoniously alongside `TrainerFn` to provide detailed context about the ongoing operations.
#
# Values:
#       -  Training: Indicates the training phase of the model.
#       -  Sanity_checking: Denotes the preliminary phase before training, ensuring the soundness of the system.
#       -  Validating: Represents the validation phase during or after training.
#       -  Testing: Highlights the post-training evaluation phase.
#       -  Predicting: Signifies the phase where the model generates predictions on new data.
#
# Properties:
#       -  evaluating: A boolean property that returns True if the current stage involves model evaluation, i.e., during validation, 
#                      testing, or sanity checking.
#       -  dataloader_prefix: A utility property providing a simplified prefix based on the current stage, notably useful for dataloader 
#                             operations and naming conventions.
#
# Design Philosophy:
#       1. Contextual Richness: By offering a variety of stages beyond the primary functions, the enum lends rich context, 
#                               facilitating nuanced logic or event-driven mechanisms in training workflows.
#
#       2. Compatibility with TrainerFn: `RunningStage` works in tandem with `TrainerFn`, complementing the latter's function-centric 
#                                        mapping with stage-focused delineation.
#
#       3. Utility-Driven Properties: With properties like `evaluating` and `dataloader_prefix`, the enum isn't just a passive 
#                                      classification; it actively assists in the streamlining of code logic.
#
#       4. Flexibility & Precision: The enumeration allows for multiple running stages during a single `TrainerFn`, 
#                                   granting developers greater flexibility and precision in their operations.
#
# The `RunningStage` enum plays a pivotal role in portraying the intricate stages of the training process within "FlameModules". 
# It upholds the framework's emphasis on clarity, granularity, and developer empowerment, ensuring that users can accurately discern 
# and react to the various stages of model training and evaluation.

class RunningStage(LightningEnum):
    #@: Enum for the current running stage.
    #@: This stage complements `TrainerFn` by specifying the current running stage for each function. 
    #@: More than one running stage value can be set while a `TrainerFn` is running. 
    #
    #   - `TrainerFn.Fitting` : `RunningStage.{Sanity_checking, Training, Validating}`
    #   - `TrainerFn.Validating` : `RunningStage.Validating`
    #   - `TrainerFn.Testing` : `RunningStage.Testing`
    #   - `TrainerFn.Predicting` : `RunningStage.Predicting`
    #    
    Training: str = 'train'
    Sanity_checking: str = 'sanity_check'
    Validating: str = 'validate'
    Testing: str = 'test'
    Predicting: str = 'predict'
    
    @property
    def evaluating(self) -> bool:
        return self in (self.Validating, self.Testing, self.Sanity_checking)
    
    
    @property
    def dataloader_prefix(self) -> Optional[str]:
        if self in (self.Validating, self.Sanity_checking):
            return 'val'
        
        return self.value
    
    
# Class TrainerState:
# In the "FlameModules" ecosystem, the `TrainerState` serves as a compact representation of the trainer's current operational state, 
# bundling together various status parameters into a single cohesive unit.
#
# Attributes:
#       -  status: Represents the high-level operational status of the trainer. Initialized to 'Initializing' by default.
#       -  fn: Denotes the current user-facing function that the trainer is executing, such as 'fit', 'test', etc. Initialized to None.
#       -  stage: Signifies the specific running stage associated with the current function, offering a more granular status than 'fn'.
#                 Initialized to None.
#
# Properties:
#       -  finished: A boolean property that returns True if the trainer's status indicates completion of its task.
#       -  stopped: A boolean property that checks if the trainer's operation has ceased, either due to completion or interruption.
#
# Design Philosophy:
#
#       1. Cohesive Representation: By merging status, function, and stage into a singular entity, the dataclass provides a centralized 
#                                   view of the trainer's state, obviating the need for scattered variables or flags.
#
#       2. Intuitive Status Checks: With properties like 'finished' and 'stopped', developers can quickly gauge the operational state 
#                                   without delving into attribute nuances.
#
#       3. Compatibility with Enum Classes: `TrainerState` seamlessly integrates with previously discussed enums (`TrainerStatus`, 
#                                           `TrainerFn`, and `RunningStage`), further solidifying its role as the central state manager.
#
#       4. Simplicity & Clarity: While encapsulating multifaceted trainer dynamics, the dataclass remains straightforward and intuitive, 
#                                 fostering ease of use and implementation.
#
# The `TrainerState` stands as a testament to "FlameModules" emphasis on organized, clear, and efficient code structures. It epitomizes 
# the idea of modular design, ensuring that developers can access, modify, and interpret the trainer's state with minimal hassle and 
# maximal clarity.

@dataclass
class TrainerState:
    #@: Dataclass to encapsulate the current `flame_modules.trainer.trainer.Trainer` state.
    status: TrainerStatus = TrainerStatus.Initializing
    fn: Optional[TrainerFn] = None
    stage: Optional[RunningStage] = None
    
    @property
    def finished(self) -> bool:
        return self.status == TrainerStatus.Finished
    
    
    @property
    def stopped(self) -> bool:
        return self.status.stopper
    
    
#@: NOTE : SOME EX : Look : https://github.com/rahul1-bot/Rahul_PyTorch_SpecialCode/blob/main/Depth%20Estimation/DepthEstimation_trainer.py
# class DepthEstimationTrainer:
#     def __init__(self, ...) -> None:  # Existing parameters
#         # Existing initializations
#
#         # Initialize trainer status to INITIALIZING
#         self.status: str = TrainerStatus.Initializing
#
#     def train(self, epochs: int) -> Any:
#         try:
#             self.status = TrainerStatus.Running
#
#             # Existing training code
#
#             self.status = TrainerStatus.Finished
#         except:
#             self.status = TrainerStatus.Interrupted
#             raise
#
#     # Rest of your DepthEstimationTrainer methods remain unchanged
#
# trainer = DepthEstimationTrainer(...)
# print(trainer.status)  # Should print "Initializing"
# trainer.train(10)
# print(trainer.status)  # Should print "Finished" if completed successfully or "Interrupted" if there was an error.
#
# Or include
# if trainer.status.stopped:
#     print("Training has stopped.")
#
