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
from dataclasses import asdict, dataclass, field
from typing import Type

# Class 1 : {_BaseProgress} : 
# This class is designed as a utility for managing the state of certain data-driven objects.
#     *   Purpose:
#             It's a mixin, a type of utility class in object-oriented programming that provides methods for other classes without 
#             being considered as one of the base classes. The primary goal here is to manage and manipulate the internal state of 
#             data classes in a structured manner.

#     *   Methods:

#             * state_dict: Returns a dictionary representation of the instance's state. It takes advantage of the asdict function 
#                           from the dataclasses module, which is specially designed for this purpose with dataclasses.

#             * load_state_dict: Accepts a dictionary (state_dict) and updates the instance's state using the values from this 
#                                dictionary. Essentially, it overwrites the current state with the provided state.

#             * from_state_dict (Class Method): This method is a class-based factory. It creates an instance of the class, loads 
#                                               the given state into this instance, and then returns the instance with the loaded 
#                                               state. It's an alternative to the traditional instantiation and then loading the 
#                                               state in two steps.

#             * reset: This method is designed to reset the state of the instance. However, it raises a NotImplementedError, 
#                      meaning that any subclass inheriting from _BaseProgress should provide its own implementation of the reset logic.

#     Decorator @dataclass: By using this decorator, the class is defined as a dataclass. A dataclass automatically generates special 
#     methods like __init__, __repr__, and others based on the annotations and default values provided in the class body.

@dataclass
class _BaseProgress:
    #@: Mixin that implements state-loading utilities for dataclasses.
    def state_dict(self) -> dict[Any, Any]:
        return asdict(self)
    
    
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        self.__dict__.update(state_dict)
        
        
    @classmethod
    def from_state_dict(cls, state_dict: dict[Any, Any]) -> '_BaseProgress':
        obj: _BaseProgress = cls()
        obj.load_state_dict(state_dict)
        return obj
    
    
    def reset(self) -> None:
        raise NotImplementedError
    


# Class 2 : {_ReadyCompletedTracker} : 
# The _ReadyCompletedTracker class, derived from _BaseProgress, is built to monitor the progress of certain events. 
# This class has two primary attributes:

#     *   ready: It represents the number of events that are prepared to start.
    
#     *   completed: It indicates the number of events that have finished.

# The class also has two methods:
#     * reset: Restores both ready and completed attributes to their initial values, essentially setting the progress to the starting point.
    
#     * reset_on_restart: In cases where the progress needs to be restarted from where it left off, this method sets the ready attribute to 
#                         match the completed value, effectively marking all completed events as ready to be initiated again.

# This utility class is ideal for situations where it's crucial to track progress and occasionally revert or restart based on certain conditions.

@dataclass
class _ReadyCompletedTracker(_BaseProgress):
    #@: Track an event's progress.
    # Args:
    #   * ready (int): Intended to track the number of events ready to start
    #   * completed (int): Intended to be incremented after the event completed
    
    ready: Optional[int] = 0
    completed: Optional[int] = 0
    
    def reset(self) -> None:
        #@: Reset the state 
        self.ready = 0
        self.completed = 0
        
    
    def reset_on_restart(self) -> None:
        #@: Reset the progress on restart 
        self.ready = self.completed
        
        
        
# Class 3 : {_StartedTracker} :
# The _StartedTracker class is an extension of the _ReadyCompletedTracker class, designed to further refine event tracking by including an 
# additional stage of the event lifecycle: when the event has started.

# Key elements of the _StartedTracker class are:

#     *   started: An attribute that records the number of events that have initiated. This is in addition to the ready and completed 
#                  attributes inherited from the parent class.

# Methods included:

#     *   reset: This method reinitializes all attributes to their default states. It first calls the reset method of its parent class
#                to set ready and completed to their default values and then sets the started attribute to zero.

#     *   reset_on_restart: It's used to reset the progress when restarting from the last stopped point. The method first invokes the 
#                           reset_on_restart from its parent class, then equates the started attribute to the completed attribute. 
#                           This implies that all events that were completed are now set to have started again.

# In essence, the _StartedTracker provides a more granular level of progress tracking, specifically accounting for events that have been 
# initiated but not necessarily completed.
@dataclass
class _StartedTracker(_ReadyCompletedTracker):
    #@: Track an event's progress. 
    # Args:
    #   * ready (int): Intended to track the number of events ready to start
    #   * started (int): Intented to be incremented after the event is started 
    #   * completed (int): Intended to be incremented after the event completes
    
    started: Optional[int] = 0
    
    def reset(self) -> None:
        super(_StartedTracker, self).reset()
        self.started = 0
        
        
    def reset_on_restart(self) -> None:
        super(_StartedTracker, self).reset_on_restart()
        self.started = self.completed
        
        
# Class 4 : {_ProcessedTracker} :
# The _ProcessedTracker class is an augmentation of the _StartedTracker class, further extending the event tracking capabilities by adding an 
# additional state, processed, to monitor events that have been acted upon but not yet concluded.

# Key highlights of the _ProcessedTracker class:

#     *   processed: An attribute that tallies the number of events that have been processed. It accompanies other attributes like ready, 
#                    started, and completed to offer a comprehensive event lifecycle tracking.

# Methods in the class:

#     *   reset: Resets all the attributes to their foundational states. It invokes the reset method of the parent (_StartedTracker) to 
#                revert ready, started, and completed to their initial values. Subsequently, it resets the processed attribute to zero.

#     *   reset_on_restart: Used to restore the progress when recommencing from a previously halted point. This method initially uses the 
#                           parent class's reset_on_restart to set the ready and completed attributes appropriately. Then, the processed 
#                           attribute is aligned with the completed attribute, signifying that all completed events are considered as processed again.

# Overall, the _ProcessedTracker class enriches the progress tracking spectrum by distinguishing between events that are in the midst of processing 
# and those that have concluded, providing a clearer perspective of the event lifecycle.
@dataclass
class _ProcessedTracker(_StartedTracker):
    #@: Track an event's progress. 
    # Args:
    #   * ready (int): Intended to track the number of events ready to start
    #   * started (int): Intended to be incremented after the event is started 
    #   * processed (int): Intended to be incremented after the event is processed
    #   * completed (int): Intended to be incremented after the event complete 
    
    processed: Optional[int] = 0
    
    def reset(self) -> None:
        super(_ProcessedTracker, self).reset()
        self.processed = 0
        
    
    def reset_on_restart(self) -> None:
        super(_ProcessedTracker, self).reset_on_restart()
        self.processed = self.completed
         
    
    
    
# Class 5 : {_Progress} :
# The _Progress class is an advanced tracking module designed to monitor both the aggregate and the immediate progress of events. It leverages 
# the previously discussed _ReadyCompletedTracker and its derivatives for comprehensive progress management.

# Main Features of the _Progress class:

#     * Attributes:
#         *   total: Represents the cumulative progress of events. It is an instance of the _ReadyCompletedTracker or its descendants.

#         *   current: Chronicles the ongoing event's progress using an identical tracker as total.

    
#     * Initialization and Checks:
#         *   __post_init__: Ensures that both total and current trackers are of the same type, ensuring synchronized tracking.


#     * Increment Methods:
#         *   These functions (increment_ready, increment_started, increment_processed, increment_completed) increment the respective 
#             attributes in both the total and current trackers. Before making increments, checks are performed to validate if the 
#             appropriate attributes exist in the tracker.


#     * Utility Creation Method:
#         *   from_defaults: Offers a convenient way to instantiate a _Progress object by providing a tracker class and optional 
#             keyword arguments that get passed to both the total and current trackers.

#     * Reset Methods:
#         *   reset: Resets both the total and current trackers.

#         *   reset_on_run: Resets only the current tracker.

#         *   reset_on_restart: Uses the reset_on_restart method of the current tracker.

    
#     * State Management:
#         *   load_state_dict: Allows restoring the state of the total and current trackers from a given dictionary, which can be 
#                              useful for resuming from a saved state
    
@dataclass
class _Progress(_BaseProgress):
    #@: Track aggregated and current progress.
    # Args:
    #   * total (_ReadyCompletedTracker): Intended to track the total progress of an event
    #   * current (_ReadyCompletedTracker): Intended to track the current progress of an event
    
    total: _ReadyCompletedTracker = field(default_factory= _ProcessedTracker)
    current: _ReadyCompletedTracker = field(default_factory= _ProcessedTracker)
    
    def __post_init__(self) -> None:
        if self.total.__class__ is not self.current.__class__:
            raise ValueError('`total` and `current` instances should be of the same class')


    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1
    
    
    def increment_started(self) -> None:
        if not isinstance(self.total, _StartedTracker):
            raise TypeError(f'`{self.total.__class__.__name__}` does not have a `started` attribute')
        
        self.total.started += 1
        self.current.started += 1

    
    def increment_processed(self) -> None:
        if not isinstance(self.total, _ProcessedTracker):
            raise TypeError(f'`{self.total.__class__.__name__}` does not have a `processed` attribute')
        
        self.total.processed += 1
        self.current.processed += 1
        
        
    
    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1
        
    
    @classmethod
    def from_defaults(cls, tracker_cls: Type[_ReadyCompletedTracker], **kwargs: int) -> '_Progress':
        #@: Utility function to easily create an instance from keyword arguments to both ``Tracker``s
        return cls(
            total= tracker_cls(**kwargs),
            current= tracker_cls(**kwargs)
        )
        
        
    def reset(self) -> None:
        self.total.reset()
        self.current.reset()
        
        
    def reset_on_run(self) -> None:
        self.current.reset()
        
        
    def reset_on_restart(self) -> None:
        self.current.reset_on_restart()
        
        
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        self.total.load_state_dict(state_dict['total'])
        self.current.load_state_dict(state_dict['current'])
        
        


# Class 6 : {_BatchProgress} :
# The _BatchProgress class is a specialized module tailored to track the progression of batches. Built on top of the previously discussed 
# _Progress class, it incorporates an added feature to determine if a batch is the last in the sequence, especially useful for iterable datasets.

# Key Aspects of the _BatchProgress class:

#     * Attribute:
#         *   is_last_batch: A boolean attribute that indicates if the current batch being processed is the last one.


#     * Reset Methods:
#         *   reset: This function reinitializes the progress tracking, utilizing the reset capability of its parent _Progress class and 
#                    also setting the is_last_batch attribute to False.

#         *   reset_on_run: Similar to the reset function but focuses on the current progress tracking of the _Progress class.


#     * State Management:
#         *   load_state_dict: This function is employed to reinstate the state of the _BatchProgress object. It leverages the load_state_dict 
#                              function from its parent class for core attributes and separately updates the is_last_batch attribute from the 
#                              provided state dictionary.

# In summary, _BatchProgress offers specialized tracking for batches, making it suitable for environments where batch-wise processing is essential, 
# like in deep learning training loops. Its capability to recognize the last batch is particularly beneficial for datasets that are iterable, 
# ensuring proper handling and potential optimizations for the final batch.
@dataclass
class _BatchProgress(_Progress):
    #@: Tracks the progress of the batch
    # Args:
    #   *   total (int): Tracks the total batch progress
    #   *   current (int): Tracks the current batch progress
    #   *   is_last_batch (bool): Whether the batch is the last one. This is useful for iterable datasets.
    
    is_last_batch: Optional[bool] = False
    
    def reset(self) -> None:
        super(_BatchProgress, self).reset()
        self.is_last_batch = False
        
    
    def reset_on_run(self) -> None:
        super(_BatchProgress, self).reset_on_run()
        self.is_last_batch = False
        
    
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        super(_BatchProgress, self).load_state_dict(state_dict)
        self.is_last_batch = state_dict['is_last_batch']
        
        
        
        
# Class 7 : {_SchedulerProgress} :
# The _SchedulerProgress class is designed to monitor the progression of schedulers, which are often used in optimization and training loops 
# to control learning rates or other hyperparameters. This class extends the capabilities of the _Progress class.

# Key Elements of the _SchedulerProgress class:

#     * Attributes:
#         *   total: It's an instance of the _ReadyCompletedTracker class. It's designed to track the overall progress of the 
#                    scheduler throughout its lifecycle.

#         *   current: Another instance of the _ReadyCompletedTracker class, this is intended to keep a tab on the current state 
#                      of the scheduler in a given phase or cycle.


#     * Dataclass Foundations: Utilizing the Python dataclass framework, the class ensures that its attributes are automatically managed, 
#                              streamlining the instantiation process and providing utility functions by default.

# In essence, _SchedulerProgress is a tailored module to oversee the progression of schedulers. By inheriting from the _Progress class, it 
# benefits from the fundamental progress tracking functionalities while having its attributes specifically set for scheduler-centric tasks.

@dataclass
class _SchedulerProgress(_Progress):
    #@: Tracks the progress of the scheduler
    #@: These counters local to a trainer rank
    # Args:
    #   *   total (int): Tracks the total scheduler progress
    #   *   current (int): Tracks the current scheduler progress
    
    total: _ReadyCompletedTracker = field(default_factory= _ReadyCompletedTracker)
    current: _ReadyCompletedTracker = field(default_factory= _ReadyCompletedTracker)
    



# Class 8 : {_OptimizerProgress} :
# The _OptimizerProgress class, extending from _BaseProgress, is crafted to monitor optimization activities.

# Key Elements:
#     * Attributes:
#         *   step: Monitors calls to optimizer.step, using the _Progress class.

#         *   zero_grad: Tracks optimizer.zero_grad calls, also with _Progress.

    
#     * Dataclass Base: Utilizes the Python dataclass for easy instantiation and attribute management.


#     * Methods:
#         *   reset(): Resets tracking for both attributes.

#         *   reset_on_run() & reset_on_restart(): Specialized reset methods for specific scenarios.

#         *   load_state_dict(state_dict): Restores progress from a provided dictionary.

# In essence, _OptimizerProgress oversees optimization tasks, focusing on gradient application and resets.
@dataclass
class _OptimizerProgress(_BaseProgress):
    #@: Tracks the progress of the optimizer
    # Args:
    #   *   step (_Progress): Tracks the `optimizer.step` calls
    #   *   zer_grad (_Progress): Tracks the `optimizer.zero_grad` calls
    
    step: _Progress = field(default_factory= lambda: _Progress.from_defaults(_ReadyCompletedTracker))
    zero_grad: _Progress = field(default_factory= lambda: _Progress.from_defaults(_StartedTracker))
        
    
    def reset(self) -> None:
        self.step.reset()
        self.zero_grad.reset()
        
    
    def reset_on_run(self) -> None:
        self.step.reset_on_run()
        self.zero_grad.reset_on_run()
        
        
    def reset_on_restart(self) -> None:
        self.step.reset_on_restart()
        self.zero_grad.reset_on_restart()
        
    
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        self.step.load_state_dict(state_dict['step'])
        self.zero_grad.load_state_dict(state_dict['zero_grad'])
        
        
        

# Class 9 : {_OptimizationProgress} :
# The _OptimizationProgress class, stemming from _BaseProgress, oversees the general optimization process.
# Key Features:
#     * Attribute:
#         *   optimizer: Captures the progress of optimization tasks using the _OptimizerProgress class.

    
#     * Methods & Properties:
#         *   optimizer_steps: Retrieves the count of completed optimizer steps.

#         *   reset(), reset_on_run(), & reset_on_restart(): Methods resetting the optimizer's progress under varying circumstances.

#         *   load_state_dict(state_dict): Updates the optimizer's progress from a provided dictionary.


# Essentially, _OptimizationProgress provides tools for monitoring and managing the broader optimization journey.

@dataclass
class _OptimizationProgress(_BaseProgress):
    #@: Tracks the progress of the optimization 
    # Args:
    #   *   optimizer (_OptimizerProgress): Tracks the optimizer progress
    
    optimizer: _OptimizerProgress = field(default_factory= _OptimizerProgress)
    
    
    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed
    
    
    def reset(self) -> None:
        self.optimizer.reset()
        
        
    def reset_on_run(self) -> None:
        self.optimizer.reset_on_run()
        
        
    def reset_on_restart(self) -> None:
        self.optimizer.reset_on_restart()
        
        
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        

#@: Driver Code
if __name__.__contains__('__main__'):
    ...