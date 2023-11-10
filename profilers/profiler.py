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
import logging, os, time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, TextIO, Callable
from lightning.fabric.utilities.cloud_io import get_filesystem

log = logging.getLogger(__name__)

# ----------------------------------
# Profiler in FlameModules
# ----------------------------------
#
# Purpose:
#     Profiler is an abstract base class designed for performance monitoring in FlameModules. It provides an interface for 
#     recording the start and stop times of different actions during the training lifecycle, facilitating detailed performance 
#     analysis.
#
# Methods:
#   - `start`: Abstract method to define how to begin recording an action.
#   - `stop`: Abstract method to define how to stop recording an action and log the duration.
#   - `summary`: Summarizes the recorded profiling information.
#   - `profile`: Context manager for profiling a block of code.
#   - `describe`: Outputs the profiling summary after the training run.
#   - `setup`: Prepares the profiler for the upcoming stage.
#   - `teardown`: Cleans up after profiling is complete.
#
# Parameters:
#   - `dirpath`: Directory path for saving profiler logs.
#   - `filename`: Name of the file to save profiler logs.
#   - `action_name`: The name of the action being profiled.
#   - `extension`: File extension for profiler logs.
#   - `split_token`: Delimiter used in filenames.
#
# Key Steps:
#   1. `start` and `stop` are to be implemented by subclasses to specify profiling behavior.
#   2. `summary` typically returns a string representation of the profiled metrics.
#   3. `profile` wraps the start and stop methods for use in a `with` statement.
#   4. `describe` calls `summary` and writes the output to a file or logs it.
#   5. `setup` initializes the profiler's stage and rank information.
#   6. `teardown` finalizes the profiling process, closing any open files or streams.
#
# Importance:
#   - Profiler is essential for identifying bottlenecks and optimizing the training process within the FlameModules framework.
#   - It provides a standardized approach to measure and report on the time taken by different parts of the training loop.
#   - The abstract methods enforce a consistent interface for all subclasses, ensuring that custom profilers integrate smoothly.


class Profiler(ABC):
    def __init__(self, dirpath: Optional[Union[str, Path]] = None, filename: Optional[str] = None) -> None:
        self.dirpath = dirpath
        self.filename = filename
        
        self._output_file: Optional[TextIO] = None
        self._write_stream: Optional[Callable[Any]] = None
        self._local_rank: Optional[int] = None
        self._stage: Optional[str] = None    
        
        
        
    @abstractmethod
    def start(self, action_name: str) -> None:
        #@: Defines how to start recording an action
        ...
        
        
    @abstractmethod
    def stop(self, action_name: str) -> None:
        #@: Defines how to record the duration once an action is complete
        ...
        
        
        
    def summary(self) -> str:
        return ''
    
    
    
    @contextmanager
    def profile(self, action_name: str) -> Generator:
        #@: Yields a context manager to encapsulate the scope of a profiled action
        # The profiler will start once you've entered the context and will automatically
        # stop once you exit the code block. 
        ...
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)
            
            
            
    def _rank_zero_info(self, *args: Any, **kwargs: Any) -> None:
        if self._local_rank in [None, 0]:
            log.info(*args, **kwargs)
            
            
    
    
    def _prepare_filename(self, action_name: Optional[str] = None, extension: Optional[str] = '.txt', split_token: Optional[str] = '-') -> str:
        args: list[Any] = []
        if self._stage is not None:
            args.append(self._stage)
        if self.filename:
            args.append(self.filename)
        if self._local_rank is not None:
            args.append(str(self._local_rank))
        if action_name is not None:
            args.append(action_name)
        return split_token.join(args) + extension
    
    
    
    def _prepare_streams(self) -> None:
        if self._write_stream is not None:
            return 
        
        if self.filename and self.dirpath:
            filepath = os.path.join(self.dirpath, self._prepare_filename())
            fs = get_filesystem(filepath)
            fs.mkdirs(self.dirpath, exist_ok= True)
            file = fs.open(filepath, 'a')
            self._output_file = fole
            self._write_stream = file.write
        else:
            self._write_stream = self._rank_zero_info
            
            
    
    
    def describe(self) -> None:
        #@: Logs a profile report after the conclusion of run. 
        self._prepare_stream()
        summary = self.summary()
        if summary and self._write_stream is not None:
            self._write_stream(summary)
        if self._output_file is not None:
            self._output_file.flush()
        self.teardown(stage= self._stage)
        
        
    
    
    def _stats_to_str(self, stats: dict[str, str]) -> str:
        stage: str = f'{self._stage.upper()}' if self._stage is not None else ''
        output: list[str] = [stage + 'Profiler Report']
        for action, value in stats.items():
            header: str = f'Profile stats for : {action}'
            if self._local_rank is not None:
                header += f' rank: {self._local_rank}'
            output.append(header)
            output.append(value)
        return os.linesep.join(output)
    
    
    
    def setup(self, stage: str, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
        #@: Execute arbitrary pre-profiling set-up steps.
        self._stage = stage
        self._local_rank = local_rank
        self.dirpath = self.dirpath or log_dir
        
        
        
    def teardown(self, stage: Optional[str]) -> None:
        #@: Execute arbitrary post-profiling tear-down steps.
        self._write_stream = None
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None
            
            
            
    def __del__(self) -> None:
        self.teardown(stage= self._stage)
        
        
        
    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank    
        



#@: :: NOTE :: Example Code :: NOTE 
# def get_filesystem(path):
#     # Mock filesystem operations
#     # replace with your actual filesystem operation
#     return os


# class CustomProfiler(Profiler):
#     def __init__(self, dirpath: Optional[Union[str, Path]] = None, filename: Optional[str] = None):
#         super().__init__(dirpath, filename)
#         self._action_start_times = {}
#         self._action_durations = {}



#     def start(self, action_name: str) -> None:
#         self._action_start_times[action_name] = time.monotonic()



#     def stop(self, action_name: str) -> None:
#         start_time = self._action_start_times.pop(action_name, None)
#         if start_time:
#             elapsed_time = time.monotonic() - start_time
#             self._action_durations[action_name] = elapsed_time



#     def summary(self) -> str:
#         summary_lines = [f"{action}: {duration:.2f}s" for action, duration in self._action_durations.items()]
#         return "\n".join(summary_lines)



#     @contextmanager
#     def profile(self, action_name: str) -> Generator:
#         self.start(action_name)
#         yield
#         self.stop(action_name)



#     def _prepare_filename(self, action_name: Optional[str] = None, extension: Optional[str] = '.txt', split_token: Optional[str] = '-') -> str:
#         # Implement the filename preparation logic
#         args = [self._stage, self.filename, str(self._local_rank), action_name]
#         filename = split_token.join(filter(None, args)) + extension
#         return filename



#     def _prepare_streams(self) -> None:
#         # Implement the stream preparation logic
#         if self._write_stream is not None:
#             return

#         if self.filename and self.dirpath:
#             fs = get_filesystem(str(self.dirpath))
#             fs.mkdirs(self.dirpath, exist_ok=True)
#             filepath = os.path.join(self.dirpath, self._prepare_filename())
#             file = open(filepath, 'a')
#             self._output_file = file
#             self._write_stream = file.write
#         else:
#             self._write_stream = print  # Use print if no file output is set up



#     def describe(self) -> None:
#         # Implement the describe logic
#         self._prepare_streams()
#         summary = self.summary()
#         if summary and self._write_stream is not None:
#             self._write_stream(summary + "\n")
#         if self._output_file is not None:
#             self._output_file.flush()
#             self._output_file.close()
#             self._output_file = None



#     def setup(self, stage: str, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
#         # Implement setup logic
#         self._stage = stage
#         self._local_rank = local_rank
#         self.dirpath = self.dirpath or log_dir



#     def teardown(self, stage: Optional[str] = None) -> None:
#         # Implement teardown logic
#         self._write_stream = None
#         if self._output_file is not None:
#             self._output_file.close()
#             self._output_file = None



#     def __del__(self) -> None:
#         self.teardown(stage=self._stage)



#     @property
#     def local_rank(self) -> int:
#         return 0 if self._local_rank is None else self._local_rank


# #@: :: NOTE :: Custom Low Functionality Trainer :: NOTE 
# class Trainer:
#     def __init__(
#         self,
#         # ... other parameters ...
#         profiler: Optional[CustomProfiler] = None,
#         # ... other parameters ...
#     ) -> None:
#         # ... existing initialization code ...

#         # Initialize the profiler
#         self.profiler = profiler or CustomProfiler()
#         self.profiler.setup(stage='init', local_rank=0, log_dir='/path/to/save/profile')

#         # ... other initialization code ...


#     def train(self):
#         # Using the profiling in the training method
#         with self.profiler.profile('train'):
#             # ... training loop ...
#             pass


#     def validate(self):
#         # Example of profiling the validation step
#         with self.profiler.profile('validate'):
#             # ... validation loop ...
#             pass


#     def test(self):
#         # Example of profiling the testing step
#         with self.profiler.profile('test'):
#             # ... testing loop ...
#             pass



#     def fit(self):
#         # Fit could be the method that calls train, validate, test, etc.
#         # Start the profiler for the entire fit process
#         self.profiler.start('fit')
#         try:
#             self.train()
#             self.validate()
#             self.test()
#         finally:
#             # Stop the profiler and print out the summary
#             self.profiler.stop('fit')
#             self.profiler.describe()


# #@: Driver Code
# if __name__.__contains__('__main__'):
#     custom_profiler = CustomProfiler(dirpath='/path/to/save/profile', filename='training_profile')
#     trainer = Trainer(profiler = custom_profiler)
#     trainer.fit()
