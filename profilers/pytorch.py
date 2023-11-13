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
import inspect, logging, os, torch
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, ContextManager, Union, NewType
import torch.nn as nn

from torch.autograd.profiler import EventList, record_function
from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle

from lightning.fabric.accelerators.cuda import is_cuda_available
from FlameModules.profilers.profiler import Profiler
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn


log = logging.getLogger(__name__)
warning_cache = WarningCache()


_Profiler = NewType('_Profiler', Union[torch.profiler.profile, torch.autograd.profiler.profile, torch.autograd.profiler.emit_nvtx])
_Kineto_available = torch.profiler.kineto_available()




class RegisterRecordFunction:
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._records: dict[str, record_function] = {}
        self._handles: dict[str, list[RemovableHandle]] = {}
        
        
        
    def _start_recording_forward(self, _: nn.Module, input_tensor: torch.tensor, record_name: str) -> torch.tensor:
        record = record_function('[flame][module]' + record_name)
        record.__enter__()
        self._records[record_name] = record
        return input_tensor
    
    
    
    
    def _stop_recording_forward(self, _: nn.Module, __: torch.tensor, output_tensor: torch.tensor, record_name: str) -> torch.tensor:
        self._records[record_name].__exit__(None, None, None)
        return output_tensor
    
    
    
    
    def __enter__(self) -> None:
        for module_name, module in self._model.named_modules():
            if module_name:
                full_name: str = f'{type(module).__module__}.{type(module).__name__}'
                record_name: str = f'{full_name}: {module_name}'
                pre_forward_handle = module.register_forward_pre_hook(
                    partial(self._start_recording_forward, record_name= record_name)
                )
                post_forward_handle = module.register_forward_handle(
                    partial(self._start_recording_forward, record_name= record_name)
                )
                
                self._handles[module_name] = [pre_forward_handle, post_forward_handle]
                
        
        
                
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}
        
        
        
        
        
class ScheduleWrapper:
    #@: This class is used to override the schedule logic from the profiler and perform recording for both 
    # `training_step`, `validation_step`
    
    def __init__(self, schedule: Callable) -> None:
        if not _Kineto_available:
            raise ModuleNotFoundError(
                'You are trying to use `ScheduleWrapper` which require kineto install.'
            )
        self._schedule = schedule
        self.reset()
        
        
    
    def reset(self) -> None:
        #@: Handle properly `fast_dev_run`. Pytorch Profiler will fail otherwise
        self._num_training_step: int = 0
        self._num_validation_step: int = 0
        self._num_test_step: int = 0
        self._num_predict_step: int = 0
        self._training_step_reached_end: bool = False
        self._validation_step_reached_end: bool = False
        self._test_step_reached_end: bool = False
        self._predict_step_reached_end: bool = False
        
        #@: Used to stop profiler when `ProfilerAction.Record_and_Save` is reached
        self._current_action: Optional[str] = None
        self._prev_schedule_action: Optional[ProfilerAction] = None
        self._start_action_name: Optional[str] = None
        
        
        
    
    def setup(self, start_action_name: str) -> None:
        self._start_action_name = start_action_name
        
        
        
    def pre_step(self, current_action: str) -> None:
        self._current_action = current_action
    
    
    
    
    @property
    def is_training(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('training_step')
        
        
        
    @property
    def is_validating(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('validation_step')
        
        
        
    @property
    def is_testing(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('test_step')
        
        
        
    @property
    def is_predicting(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('predict_step')
        
        
        
    
    @property
    def num_step(self) -> int:
        if self.is_training:
            return self._num_training_step
            
        if self.is_validating:
            return self._num_validation_step
            
        if self.is_testing:
            return self._num_test_step
            
            
        if self.is_predicting:
            return self._num_predict_step
            
            
        return 0
        
        
        
        
    
    def _step(self) -> None:
        if self.is_training:
            self._num_training_step += 1
        
        elif self.is_validating:
            assert self._start_action_name is not None
            if self._start_action_name.endswith('on_fit_start'):
                if self._num_training_step > 0:
                    self._num_validation_step += 1                    
            else:
                self._num_validation_step += 1
            
        elif self.is_testing:
            self._num_test_step += 1
        
        elif self.is_predicting:
            self._num_predict_step += 1
            
            
            
    
    @property
    def has_finished(self) -> bool:
        if self.is_training:
            return self._training_step_reached_end
        if self.is_validating:
            return self._validation_step_reached_end
        if self.is_testing:
            return self._test_step_reached_end
        if self.is_predicting:
            return self._predict_step_reached_end
            
        return False
        
        



    def __call__(self, num_step: int) -> 'ProfilerAction':
        if self._current_action is None or self.has_finished:
            return ProfilerAction.NONE
            
        self._step()
        action = self._schedule(max(self.num_step, 0))    
        if self._prev_schedule_action == Profiler.Record and action == ProfilerAction.Warmup:
            action = ProfilerAction.Record
        
        if action == ProfilerAction.Record_and_Save:
            if self.is_training:
                self._training_step_reached_end = True
            elif self.is_validating:
                self._validation_step_reached_end = True
            elif self.is_testing:
                self._test_step_reached_end = True
            elif self.is_predicting:
                self._predict_step_reached_end = True
                
        self._prev_schedule_action = action
        return action
        
        
        
        
        
class PytorchProfiler(Profiler):
    step_functions: set[str] = {
        'training_step', 'validation_step', 'test_step', 'predict_step'
    }
    available_sort_keys: set[str] = {
        'cpu_time', 'cuda_time', 'cpu_time_total',
        'cuda_time_total', 'cpu_memory_usage', 'cuda_memory_usage',
        'self_cpu_memory_usage', 'self_cuda_memory_usage', 'count'
    }
    
    def __init__(self, dirpath: Optional[Union[str, Path]] = None, filename: Optional[str] = None, group_by_input_shapes: Optional[bool] = False, emit_nvtx: Optional[bool] = False, export_to_chrome: Optional[bool] = True, row_limit: Optional[int] = 20, sort_by_key: Optional[str] = None, record_module_names: Optional[bool] = True, table_kwargs: optional[dict[str, Any]] = None, **profiler_kwargs: Any) -> None:
        #@: This Profiler uses PyTorch's AutoGrad Profiler and lets you inspect the cost of different operators 
        # inside your model - both on the CPU and GPU.
        super(PytorchProfiler, self).__init__(dirpath= dirpath, filename= filename)
        self._group_by_input_shapes = group_by_input_shapes and profiler_kwargs.get('record_shapes', False)
        self._emit_nvtx = emit_nvtx
        self._export_to_chrome = export_to_chrome
        self._row_limit = row_limit
        self._sort_by_key = sort_by_key or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"
        self._record_module_names = record_module_names
        self._profiler_kwargs = profiler_kwargs
        self._table_kwargs = table_kwargs if table_kwargs is not None else {}
        
        self.profiler: Optional[_Profiler] = None
        self.function_events: Optional['EventList'] = None
        self._flame_module: Optional['flame_modules.FlameModule'] = None
        self._parent_profiler: Optional[ContextManager] = None
        self._recording_map: dict[str, record_function] = {}
        self._start_action_name: Optional[str] = None
        self._schedule: Optional[ScheduleWrapper] = None
        
        
        if _Kineto_availble:
            self._init_kineto(profiler_kwargs)
            
            
        if self._sort_by_key not in self.available_sort_keys:
            raise MisconfigurationException(
                f"Found sort_by_key: {self._sort_by_key}. Should be within {self.available_sort_keys}."
            )
        
        for key in self._table_kwargs:
            if key in {'sort_by', 'row_limit'}:
                raise KeyError(
                    f"Found invalid table_kwargs key: {key}. This is already a positional argument of the Profiler."
                )
            valid_table_keys = set(inspect.signature(EventList.table).parameters.keys()) - {
                'self', 'sort_by', 'row_limit'
            }
            if key not in valid_table_keys:
                raise KeyError(f"Found invalid table_kwargs key: {key}. Should be within {valid_table_keys}.")
                
                
    
    
    
    def _init_kineto(self, profiler_kwargs: Any) -> None:
        has_schedule = 'scheduler' in profiler_kwargs
        self._has_on_trace_ready = 'on_trace_ready' in profiler_kwargs
        
        schedule = profiler_kwargs.get('schedule', None)
        if schedule is not None:
            if not callable(schedule):
                raise MisconfigurationException(f"Schedule should be a callable. Found: {schedule}")
            action = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise MisconfigurationException(
                    f"Schedule should return a `torch.profiler.ProfilerAction`. Found: {action}"
                )

        self._default_schedule()
        schedule = schedule if has_schedule else self._default_schedule()
        self._schedule = ScheduleWrapper(schedule) if schedule is not None else schedule
        self._profiler_kwargs['schedule'] = self._schedule
        
        activities = profiler_kwargs.get('activities', None)
        self._profiler_kwargs['activities'] = activities or self._default_activities()
        self._export_to_flame_graph = profiler_kwargs.get('export_to_flame_graph', False)
        self._metric = profiler_kwargs.get('metric', 'self_cpu_time_total')
        with_stack = profiler_kwargs.get('with_stack', False) or self._export_to_flame_graph
        self._profiler_kwargs['with_stack'] = with_stack
        
        
        
    
    @property
    def _total_steps(self) -> Union[int, float]:
        assert self._schedule is not None
        assert self._flame_module is not None
        trainer: 'flame_modules.Trainer' = self._flame_module.trainer
        
        if self._schedule.is_training:
            return trainer.num_training_batches
            
        if self._schedule.is_validating:
            num_val_batches = (
                sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
            )
            num_sanity_val_batches = (
                sum(trainer.num_sanity_val_batches) if isinstance(trainer.num_sanity_val_batches, list) else trainer.num_sanity_val_batch
            )
            return num_val_batches + num_sanity_val_batches
            
        if self._schedule.is_testing:
            num_test_batches = (
                sum(trainer.num_test_batches) if isinstance(trainer.num_test_batches, list) else trainer.num_test_batches
            )
            return num_test_batches
            
        if self._schedule.is_predicting:
            return sum(trainer.num_predict_batches)
            
        raise NotImplementedError('Unsupported Schedule')
        
        
        
    
    def _should_override_schedule(self) -> bool:
        return (
            self._flame_module is not None
            and self._schedule is not None
            and self._total_steps < 5
            and self._schedule._schedule == self._default_schedule()
        )
        
        
        
    @staticmethod
    @lru_cache(1)
    def _default_schedule() -> Optional[Callable]:
        if _Kineto_available:
            return torch.profiler.schedule(wait= 1, warmup= 1, active= 3)
        return None
        
        
        
    
    def default_activities(self) -> list['ProfilerActivity']:
        activities: list['ProfilerActivity'] = []
        if not _Kineto_available:
            return activities
        if self._profiler_kwargs.get('use_cpu', True):
            activities.append(ProfilerActivity.CPU)
        if self._profiler_kwargs.get('use_cuda', is_cuda_available()):
            activities.append(ProfilerActivity.CUDA)
        return activities
        
        
        
    
    
    def start(self, action_name: str) -> None:
        if self.profiler is None:
            if torch.autograd._profiler_enabled():
                torch.autograd._disable_profiler()
                
            if self._schedule is not None:
                self._schedule.setup(action_name)
                
            self._create_profilers()
            
            profiler = self.profiler.__enter__()
            if profiler is not None:
                self.profiler = profiler
                
            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()
        
        if self._flame_module is not None and self._register is None and self._record_module_names:
            self._register = RegisterRecordFunction(self._flame_module)
            self._register.__enter__()
            
            
        if self.profiler is not None and action_name not in self._recording_map:
            recording = record_function('[flame][profile]' + action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording
            
            
            
    
    def step(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]
            
        if not _Kineto_available or self._emit_nvtx:
            return 
            
        if self.profiler is not None and any(action_name.endswith(func) for func in self.step_functions):
            assert isinstance(self.profiler, torch.profiler.profile)
            if self._schedule is not None:
                self._schedule.pre_step(action_name)
                
                
            if self._should_override_schedule():
                warning_cache.warn(
                    "The PyTorch Profiler default schedule will be overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = torch.profiler.profiler._default_schedule_fn
                
            
            
            def on_trace_ready(profiler: _Profiler) -> None:
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        handler = tensorboard_trace_handler(
                            str(self.dirpath), self._prepare_filename(action_name= action_name, extension= "")
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        path = os.path.join(
                            self.dirpath, self._prepare_filename(action_name= action_name, extension= ".stack")
                        )
                        assert isinstance(profiler, torch.autograd.profiler.profile)
                        profiler.export_stacks(path, metric= self._metric)
                else:
                    rank_zero_warn("The PyTorchProfiler failed to export trace as `dirpath` is None")
                
                
                
            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready
                
            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()
            self.profiler.add_metadata('Flameword', 'FlameModules')
   
   
    
    
    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""

        if self._export_to_chrome and not _Kineto_available:
            filename = f"{self.local_rank}_trace.json"
            path_to_trace = filename if self.dirpath is None else os.path.join(self.dirpath, filename)
            self.function_events.export_chrome_trace(path_to_trace)

        data = self.function_events.key_averages(group_by_input_shapes= self._group_by_input_shapes)
        table = data.table(sort_by= self._sort_by_key, row_limit= self._row_limit, **self._table_kwargs)

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)
    
    
    
    
    def _create_profilers(self) -> None:
        if self.profiler is not None:
            return

        if self._emit_nvtx:
            if self._parent_profiler is None:
                self._parent_profiler = torch.cuda.profiler.profile()
            self.profiler = self._create_profiler(torch.autograd.profiler.emit_nvtx)
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(
                torch.profiler.profile if _KINETO_AVAILABLE else torch.autograd.profiler.profile
            )
            
        
        
    
    def _create_profiler(self, profiler: Type[_Profiler]) -> _Profiler:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs: dict = {
            k: v for k, v in self._profiler_kwargs.items() if k in init_parameters
        }
        return profiler(**kwargs)            
                
            
            
    
    def _cache_functions_events(self) -> None:
        if self._emit_nvtx:
            return

        if _Kineto_available:
            assert isinstance(self.profiler, torch.profiler.profile)
            self.function_events = self.profiler.events()
        else:
            assert isinstance(self.profiler, torch.autograd.profiler.profile)
            self.function_events = self.profiler.function_events
        
    
    
    
    def _delete_profilers(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self._cache_functions_events()
            self.profiler = None

        if self._schedule is not None:
            self._schedule.reset()

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None
            
            
    
    
    def teardown(self, stage: Optional[str]) -> None:
        self._delete_profilers()

        for k in list(self._recording_map):
            self.stop(k)
        self._recording_map = {}

        super().teardown(stage=stage)



#@: Driver Code
if __name__.__contains__('__main__'):
    print('hemllo')