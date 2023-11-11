# MIT License
#     
# Copyright (c) 2023 Rahul Sawhney
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
from FlameModules.profilers.profiler import Profiler
import logging, os, time, torch
from collections import defaultdict
from pathlib import Path
from typing import NewType, Tuple, List
log = logging.getLogger(__name__)


_table_row_extended = NewType('_table_row_extended', Tuple[str, float, int, float, float])
_table_data_extended = NewType('_table_data_extended', List[_table_row_extended])
_table_row = NewType('_table_row', Tuple[str, float, float])
_table_data = NewType('_table_data', List[_table_row])




class SimpleProfiler(Profiler):
    # SimpleProfiler: 
    #     A Profiler Implementation for FlameModules
    #
    # Purpose:
    #     To record the duration of various actions during training and provide a report on their
    #     average duration, total time, and percentage of total training time.
    #
    # Parameters:
    #   - `dirpath`: Directory path for saving the profiling report.
    #   - `filename`: Name of the file to save the profiling report.
    #   - `extended`: Flag to determine whether to produce an extended report with more details.
    #
    # Attributes:
    #   - `current_actions`: Dictionary to keep track of start times for active actions.
    #   - `recorded_durations`: Default dictionary to store lists of durations for each action.
    #   - `extended`: Boolean indicating if the extended report should be generated.
    #   - `start_time`: Time when the profiler was initialized.
    #
    # Methods:
    #   - `start`: Begins recording time for a given action.
    #   - `stop`: Stops recording time for a given action and stores the duration.
    #   - `_make_report_extended`: Generates a detailed report of the profiled actions.
    #   - `_make_report`: Generates a summary report of the profiled actions.
    #   - `summary`: Returns a string representation of the profiling report.
    #   - `log_row_extended`: Static method for formatting a row in the extended report.
    #   - `log_row`: Static method for formatting a row in the summary report.
    #
    # Usage:
    #   - Instantiate `SimpleProfiler` before training begins.
    #   - Use `start` and `stop` methods to profile specific actions.
    #   - Call `summary` after training to get a report of all profiled actions.
    #
    # This profiler helps identify bottlenecks in the training process and provides insights for optimization.
        
    def __init__(self, dirpath: Optional[Union[str, Path]] = None, filename: Optional[str] = None, extended: Optional[bool] = True) -> None:
        super(SimpleProfiler, self).__init__(dirpath= dirpath, filename= filename)
        self.current_actions: dict[str, float] = {}
        self.recorded_durations: dict[Any, list] = defaultdict(list)
        self.extended = extended
        self.start_time = time.monotonic()
        
        
        
    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(f'Attempted to start {action_name} which has already started.')
        self.current_actions[action_name] = time.monotonic()
        
        
    
    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(f'Attempting to stop recording an action ({action_name}) which was never started.')
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

        
        
    def _make_report_extended(self) -> tuple(_table_data_extended, float, float):
        total_duration = time.monotonic() - self.start_time
        report: list[Any] = []
        
        for a, d in self.recorded_durations.items():
            d_tensor: torch.tensor = torch.tensor(d)
            len_d: int = len(d)
            sum_d = torch.sum(d_tensor).item()
            percentage_d = 100.0 * sum_d / total_duration
            report.append(
                (a, sum_d/len_d, len_d, sum_d, percentage_d) 
            )
        
        report.sort(key= lambda x: x[4], reverse= True)
        total_calls = sum(x[2] for x in report)
        return report, total_calls, total_duration
        
    
    
    def _make_report(self) -> _table_data:
        report: list[Any] = []
        for action, d in self.recorded_durations.items():
            d_tensor: torch.tensor = torch.tensor(d)
            sum_d = torch.sum(d_tensor).item()
            report.append(
                (action, sum_d/len(d), sum_d)
            )
        
        report.sort(key= lambda x: x[1], reverse= True)
        return report
    
    
    
    
    def summary(self) -> str:
        sep = os.linesep
        output_string: str = ''
        if self._stage is not None:
            output_string += f'{self._stage.upper()}'
        output_string += f'Profiler Report{sep}'
        
        if self.extended:
            if len(self.recorded_durations) > 0:
                max_key = max(len(k) for k in self.recorded_durations)
                
                header_string: str = SimpleProfiler.log_row_extended(
                    'Action', 'Mean duration (s)', 'Num calls', 'Total time (s)', 'Percentage %'
                )
                
                output_string_len: int = len(header_string.expandtabs())
                sep_lines = f"{sep}{'-' * output_string_len}"
                output_string += sep_lines + header_string + sep_lines
                report_extended, total_calls, total_duration = self._make_report_extended()
                output_string += SimpleProfiler.log_row_extended(
                    'Total', '-', f'{total_calls:}', f'{total_duration:.5}', '100 %'
                )
                output_string += sep_lines
                for action, mean_duration, num_calls, total_duration, duration_per in report_extended:
                    output_string += SimpleProfiler.log_row_extended(
                        action,
                        f'{mean_duration:.5}',
                        f'{num_calls}',
                        f'{total_duration:.5}',
                        f'{duration_per:.5}',
                    )
                output_string += sep_lines
                    
        else:
            max_key = max(len(k) for k in self.recorded_durations)
            header_string = SimpleProfiler.log_row('Action', 'Mean duration (s)', 'Total time (s)')
            utput_string_len = len(header_string.expandtabs())
            sep_lines = f"{sep}{'-' * output_string_len}"
            output_string += sep_lines + header_string + sep_lines
            report = self._make_report()
            
            for action, mean_duration, total_duration in report:
                output_string += SimpleProfiler.log_row(action, f'{mean_duration:.5}', f'{total_duration:.5}')
            output_string += sep_lines
        
        output_string += sep
        return output_string
            
            
    
    
    @staticmethod
    def log_row_extended(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
        row: str = f'{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|'
        row += f'  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|'
        return row
        
    
    
    @staticmethod
    def log_row(action: str, mean: str, total: str) -> str:
        return f'{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|  {total:<15}\t|'
        
    
    
#@: :: NOTE :: Code Example :: NOTE
# class Trainer:
#     # ... [other parts of the Trainer class] ...

#     def __init__(
#         self,
#         # ... [other parameters] ...
#         profiler: Optional[Union[Profiler, str]] = None,
#         # ... [other parameters] ...
#     ) -> None:
#         # ... [existing code] ...
#         self.profiler = profiler
#         # ... [existing code] ...

#         # If a profiler is provided, set it up here or in a method that gets called before training starts
#         if self.profiler:
#             self._setup_profiler()

#     def _setup_profiler(self):
#         # Infer local_rank and log_dir or set defaults here
#         local_rank = getattr(self, "local_rank", 0)
#         log_dir = getattr(self, "default_root_dir", None)
        
#         # Now call the profiler's setup method with the inferred or default values
#         self.profiler.setup(stage='train', local_rank=local_rank, log_dir=log_dir)

#     def train(self):
#         # Start the overall training profiling
#         if self.profiler:
#             self.profiler.start("total_training")

#         try:
#             # ... [training loop logic here] ...
#             pass
#         finally:
#             # Stop the overall training profiling and print summary
#             if self.profiler:
#                 self.profiler.stop("total_training")
#                 self.profiler.describe()
#                 self.profiler.teardown(stage='train')



    
#@: Driver Code
# if __name__.__contains__('__main__'):
    # profiler = SimpleProfiler(
        # dirpath="path/to/profiler/reports", 
        # filename="training_profile"
    # )
    # trainer = Trainer(profiler=profiler)
    # trainer.train()