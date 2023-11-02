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
import logging, time
from FlameModules.trainer.states import RunningStage
from FlameModules.callbacks.callback import Callback
from lightning.pytorch.utilities import LightningEnum
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info

log = logging.getLogger(__name__)

class Interval(LightningEnum):
    step: str = 'step'
    epoch: str = 'epoch'
    
# ----------------------------------
# Timer in FlameModules
# ----------------------------------
# What is Timer?
#
#     Timer is a meticulously designed callback utility tailored for FlameModules, an avant-garde deep learning framework. 
#     It serves the paramount role of overseeing the temporal aspects of training, validation, and testing phases. 
#     Furthermore, Timer offers the capability to not only monitor the elapsed time but also to curtail the training 
#     procedure if it surpasses a predetermined temporal threshold, thereby ensuring time-efficient model training.
#
# Why do we need Timer in FlameModules?
#
#     1. Precision Time Oversight: With Timer, users gain an automated mechanism to precisely track the time consumed 
#                                  during various training checkpoints, eliminating the manual logging inconsistencies.
#
#     2. Adherence to Schedules: In scenarios where time constraints are stringent due to shared resources or 
#                                impending deadlines, Timer ensures the process remains within the stipulated time frame.
#
#     3. Resource Conservation: By preventing over-extended training sessions, especially in cost-intensive 
#                               computational environments, Timer plays a pivotal role in evading unnecessary financial overheads.
#
#     4. Heightened Productivity: With the automatic time surveillance and potential training halts executed by Timer, 
#                                 developers and researchers can allocate their attention to other pressing endeavors.
#
#     5. Experimental Flexibility: Researchers conducting sequential experiments can leverage Timer to allot distinct 
#                                  time slots, guaranteeing an equitable distribution of computational time among experiments.
#
# How to use Timer in FlameModules?
#
#     To weave Timer into the training fabric, it's imperative to instantiate it with a specified 'duration', 
#     symbolizing the total time allotted for the training. Once instantiated, Timer seamlessly integrates as 
#     a callback during the FlameModules training setup. As the training unfolds, Timer remains vigilant, 
#     monitoring the time and making decisions based on the user-defined constraints.

class Timer(Callback):
    #@: The `Timer` callback tracks the time spent in the training, validation and test loop and interrupts 
    # the `Trainer` if the given time limit for the training loop is reached.
    #
    # Code Doc
    #       import FLameModules as flame_modules
    #       from flame_modules.trainer.trainer import Trainer
    #       from flame_modules.callbacks.timer import Timer
    #
    #       if __name__.__contains__('__main__'):
    #           timer = Timer(duration= '00:12:00:00')
    #           trainer = Trainer(callbacks= [timer])
    #           
    #           timer.time_elapsed('train')
    #           timer.start_time('validate')
    #           timer.end_time('test)
    #
    
    def __init__(self, duration: Optional[Union[str, timedelta, dict[str, int]]] = None, interval: Optional[str] = Interval.step, 
                                                                                         verbose: Optional[bool] = True) -> None:
        # Purpose:
        #     The `Timer` callback is engineered to monitor the time spent during the training, validation, and test loops. 
        #     If a pre-specified time limit for the training loop is reached, it prompts the `Trainer` to halt the training 
        #     process. The class integrates effortlessly with the FlameModules ecosystem, offering a streamlined approach to manage 
        #     time during model training.
        #
        # Parameters:
        #   - `duration`: A parameter that can accept either a string, timedelta, or a dictionary containing days, hours, minutes, 
        #                 and seconds. This determines the maximum time allowed for the training loop.
        #   
        #   - `interval`: It specifies the frequency at which the Timer checks the time elapsed. Accepts values from a predefined 
        #                 set called `Interval`. By default, it's set to `Interval.step`.
        #
        #   - `verbose`: A boolean flag indicating whether the Timer should provide verbose outputs, keeping the user informed 
        #                about its operations. By default, it's set to True.
        #
        # Key Steps:
        #   1. Accept and process the `duration` parameter. If it's a string, it is converted to a timedelta object. If it's a 
        #      dictionary, the timedelta is constructed using the given dictionary values.
        #   
        #   2. Validate the `interval` parameter against the available options in the `Interval` set. Raise an exception for 
        #      unsupported values.
        #
        #   3. Assign the processed `duration` and `interval` values to the corresponding instance attributes `_duration` and `_interval`.
        #
        #   4. Initialize start and end time dictionaries with running stages as keys, and set their values to None. These are later 
        #      used to track time for different stages of training.
        #   
        #   5. Set an offset attribute to 0, which might be used to account for time discrepancies or adjustments during the training process.
                
        super(Timer, self).__init__()
        if isinstance(duration, str):
            dhms = duration.strip().split(':')
            dhms: list[int] = [int(item) for item in dhms]
            duration = timedelta(days= dhms[0], hours= dhms[1], minutes= dhms[2], seconds= dhms[3])
            
        if isinstance(duration, dict):
            duration = timedelta(**duration)
        
        if interval not in set(Interval):
            raise MisconfigurationException(
                f"Unsupported parameter value `Timer(interval={interval})`. Possible choices are:"
                f" {', '.join(set(Interval))}"
            )
        
        self._duration = duration.total_seconds() if duration is not None else None
        self._interval = interval
        self._verbose = verbose
        self._start_time = dict[RunningStage, Optional[float]] = {
            stage: None for stage in RunningStage
        }
        self._end_time = dict[RunningStage, Optional[float]] = {
            stage: None for stage in RunningStage
        }
        self._offset = 0
        
    
    
    def start_time(self, stage: Optional[str] = RunningStage.Training) -> Optional[float]:
        # Purpose:
        #     The `start_time` method retrieves the start time recorded for a specific stage of the training process.
        # 
        # Parameters:
        #   - `stage`: A string representing the stage of training for which the start time needs to be fetched. 
        #              The default stage is `RunningStage.Training`.
        # 
        # Key Steps:
        #   1. Convert the string `stage` parameter to its corresponding `RunningStage` enum representation.
        #   
        #   2. Fetch and return the start time from the `_start_time` dictionary for the specified `stage`.

        stage = RunningStage(stage)
        return self._start_time[stage]
    
    
    
    
    def end_time(self, stage: Optional[str] = RunningStage.Training) -> float:
        # Purpose:
        #     The `end_time` method retrieves the end time recorded for a particular stage of the training process.
        #
        # Parameters:
        #   - `stage`: A string denoting the training stage for which the end time needs to be retrieved. 
        #              By default, the stage is set to `RunningStage.Training`.
        #
        # Key Steps:
        #   1. Convert the string `stage` parameter into its respective `RunningStage` enum representation.
        #
        #   2. Access and return the end time from the `_end_time` dictionary for the indicated `stage`.

        stage = RunningStage(stage)
        return self._end_time[stage]
    
    
    
    def time_elapsed(self, stage: Optional[str] = RunningStage.Training) -> float:
        # Purpose:
        #     The `time_elapsed` method computes and returns the total time elapsed for a particular training stage, taking 
        #     into account any offsets that might be present.
        #
        # Parameters:
        #   - `stage`: A string representing the training stage for which the elapsed time needs to be calculated. 
        #              By default, the stage is set to `RunningStage.Training`.
        #
        # Key Steps:
        #   1. Retrieve the start time for the specified `stage` using the `start_time` method.
        #
        #   2. Fetch the end time for the specified `stage` using the `end_time` method.
        #
        #   3. Compute the offset. If the specified stage is `RunningStage.Training`, use the `_offset` attribute; 
        #      otherwise, the offset is 0.
        #
        #   4. If the start time is not available (i.e., None), return the offset.
        #
        #   5. If the end time is not available, calculate the time elapsed as the difference between the current 
        #      monotonic time and the start time, and then add the offset.
        #
        #   6. If both start and end times are available, compute the elapsed time as the difference between end 
        #      and start times, and then add the offset.

        start: float = self.start_time(stage)
        end: float = self.end_time(stage)
        offset = self._offset if stage == RunningStage.Training else 0
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
    
        return end - start + offset
    
    
    
    def time_remaining(self, stage: Optional[str] = RunningStage.Training) -> Optional[float]:
        # Purpose:
        #     The `time_remaining` method calculates and returns the remaining time for a specified training stage, 
        #     considering the original duration set for the loop and the time already elapsed.
        #
        # Parameters:
        #   - `stage`: A string representing the training stage for which the remaining time needs to be determined. 
        #              By default, the stage is set to `RunningStage.Training`.
        #
        # Key Steps:
        #   1. Check if `_duration` is not None.
        #
        #   2. If `_duration` exists, compute the remaining time by subtracting the elapsed time (obtained using 
        #      the `time_elapsed` method) from `_duration`.
        #
        #   3. If `_duration` does not exist, return None as the remaining time is indeterminate.

        if self._duration is None:
            return self._duration - self.time_elapsed(stage)
        return None
    
    
    
    
    def on_train_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_train_start` method is invoked when the training process commences in the FlameModules ecosystem. 
        #     It sets the start time for the training loop, allowing the `Timer` to keep track of the elapsed time during training.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that governs the training process.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` representing the model being trained.
        #
        # Key Steps:
        #   1. Register the current monotonic time (which is unaffected by system clock adjustments) as the start 
        #      time for the training loop in the `_start_time` dictionary.

        self._start_time[RunningStage.Training] = time.monotonic()
        
    
    
    
    def on_train_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_train_end` method is activated upon the conclusion of the training phase in the FlameModules framework. 
        #     Its chief role is to note down the end time of the training loop, allowing the `Timer` callback to compute the total 
        #     duration of the training process.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that is responsible for the training process.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` which signifies the neural network model undergoing training.
        #
        # Key Steps:
        #   1. Capture the current monotonic time (a steady clock that won't be affected by system clock changes) and assign it 
        #      as the end time for the training phase within the `_end_time` dictionary.

        self._end_time[RunningStage.Training] = time.monotonic()
    
    
    
    
    def on_validation_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_validation_start` method triggers at the initiation of the validation phase in the FlameModules framework. 
        #     Its primary function is to record the commencement time of the validation loop, enabling the `Timer` callback to gauge 
        #     the total duration taken by the validation process.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that oversees the validation process.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` which represents the neural network model undergoing validation.
        #
        # Key Steps:
        #   1. Obtain the current monotonic time (a consistent clock immune to system clock adjustments) and allocate it as 
        #      the start time for the validation phase within the `_start_time` dictionary.
        
        self._start_time[RunningStage.Validating] = time.monotonic()
        
        
        
        
    
    def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_validation_end` method is invoked at the culmination of the validation phase in the FlameModules framework. 
        #     It aims to register the termination time of the validation loop, facilitating the `Timer` callback in computing 
        #     the overall duration spent during the validation process.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` responsible for the validation proceedings.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` representing the neural network model that was under validation.
        #
        # Key Steps:
        #   1. Capture the current monotonic time (a continuous clock unaffected by system clock adjustments) and assign it as 
        #      the end time for the validation phase within the `_end_time` dictionary.

        self._end_time[RunningStage.Validating] = time.monotonic()
    
    
    
    
    
    def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_test_start` method is triggered at the initiation of the testing phase within the FlameModules framework. 
        #     Its primary function is to log the starting time of the testing loop, enabling the `Timer` callback to determine 
        #     the duration spent throughout the testing procedure.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` overseeing the testing proceedings.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` signifying the neural network model undergoing testing.
        #
        # Key Steps:
        #   1. Capture the present monotonic time (a continuous clock not impacted by system clock modifications) and earmark it 
        #      as the commencement time for the testing phase inside the `_start_time` dictionary.
        
        self._start_time[RunningStage.Testing] = time.monotonic()
        
    
    
    
    def on_test_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        # Purpose:
        #     The `on_test_end` method activates when the testing phase within the FlameModules framework concludes. 
        #     It primarily documents the termination time of the testing loop. This allows the `Timer` callback to compute 
        #     the total time consumed during the testing process.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that supervised the testing operations.
        #   
        #   - `flame_module`: An instance of `flame_modules.FlameModule` representing the neural network model that underwent testing.
        #
        # Key Steps:
        #   1. Record the current monotonic time (a consistent clock unaffected by system clock alterations) and designate it 
        #      as the closing time for the testing phase within the `_end_time` dictionary.

        self._end_time[RunningStage.Testing] = time.monotonic()
    
    
    
    def on_fit_start(self, trainer: 'flame_modules.Trainer', *args: Any, **kwargs: Any) -> None:
        # Purpose:
        #     The `on_fit_start` method is initiated when the fit phase in the FlameModules framework begins. This method evaluates 
        #     the remaining time and decides whether to continue the training process based on the set `duration`.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that oversees the fit operations.
        #   
        #   - `*args`: Variable list of arguments passed to the method.
        #
        #   - `**kwargs`: Variable keyword arguments passed to the method.
        #
        # Key Steps:
        #   1. Check if the `_duration` attribute is set to `None`. If true, no further action is taken.
        #   
        #   2. If `_duration` is specified, invoke the `_check_time_remaining` method to assess the time left for the training 
        #      process and take necessary actions.

        if self._duration is None:
            return 
        self._check_time_remaining(trainer)
        
        
    
    
    def on_train_batch_end(self, trainer: 'flame_modules.Trainer', *args: Any, **kwargs: Any) -> None:
        # Purpose:
        #     The `on_train_batch_end` method is triggered at the conclusion of each training batch in the FlameModules framework. 
        #     Its primary role is to monitor the time elapsed after each batch and evaluate whether to continue the training 
        #     process based on the set `duration` and `interval`.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` responsible for the training operations.
        #   
        #   - `*args`: Variable list of arguments passed to the method.
        #
        #   - `**kwargs`: Variable keyword arguments passed to the method.
        #
        # Key Steps:
        #   1. Determine if the `interval` is not set to `Interval.step` or if `_duration` is set to `None`. If either condition is 
        #      true, exit the method without any further action.
        #   
        #   2. If both conditions are satisfied, call the `_check_time_remaining` method to gauge the time left for the training 
        #      process and make the necessary decisions.

        if self._interval != Interval.step or self._duration is None:
            return 
        self._check_time_remaining(trainer)
        
    
    
    def state_dict(self) -> dict[str, dict[Any, Any]]:
        return {
            'time_elapsed': {stage.value : self.time_elapsed(stage) for stage in RunningStage}
        }
        
    
    def load_state_dict(self, state_dict: dict[str, dict[Any, Any]]) -> None:
        time_elapsed = state_dict.get('time_elapsed', {})
        self._offset = time_elapsed.get(RunningStage.Training.value, 0)
        
    
    
    
    def _check_time_remaining(self, trainer: 'flame_modules.Trainer') -> None:
        # Purpose:
        #     The `_check_time_remaining` method serves to ascertain the time left during the training process. If the time 
        #     consumed surpasses the user-specified `duration`, it signals the `Trainer` instance of FlameModules to cease the 
        #     training operation.
        #
        # Parameters:
        #   - `trainer`: An instance of `flame_modules.Trainer` that orchestrates the training activities.
        #
        # Key Steps:
        #   1. Assert that `_duration` is not set to `None`, ensuring that there's a specified time duration to monitor.
        #   
        #   2. Calculate if the total elapsed time during training exceeds or matches the set `_duration`. If so, the `should_stop` 
        #      flag is marked as True.
        #
        #   3. Utilize the `broadcast` method from the `trainer.strategy` to ensure synchronization across potential multiple devices 
        #      or nodes.
        #
        #   4. Update the `trainer.should_stop` attribute with the `should_stop` flag. If `should_stop` is True, the trainer will be 
        #      signaled to halt the training process.
        #   
        #   5. If the `should_stop` flag is True and verbose logging is enabled (`_verbose` is True), output a message detailing 
        #      the elapsed time and notify the user about the termination of the training due to the time limit being reached.

        assert self._duration is not None
        should_stop = self.time_elapsed() >= self._duration
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            elapsed = timedelta(seconds= int(self.time_elapsed(RunningStage.Training)))
            rank_zero_info(f'Time limit reaached. Elapsed time : {elapsed}. Signaling `Trainer` to stop')
            
