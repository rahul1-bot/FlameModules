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
import importlib.util, math, os, sys

if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm
    
from FlameModules.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_debug

_pad_size: int = 5



class Tqdm(_tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Tqdm, self).__init__(*args, **kwargs)
        
        
    @staticmethod
    def format_num(n: Union[int, float, str]) -> str:
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
            
        if should_be_padded and 'e' not in n:
            if '.' not in n and len(n) < _pad_size:
                try:
                    _ = float(n)
                except ValueError: return n
                n += '.'
            n += '0' * (_pad_size - len(n))
        
        return n
    
    
    

class TQDMProgressBar(ProgressBar):
    bar_formar: str = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def __init_(self, refresh_rate: Optional[int] = 1, process_position: Optional[int] = 0) -> None:
        super(TQDMProgressBar, self).__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position
        self._enabled = True
        self._train_progress_bar: Optional[_tqdm] = None
        self._val_progress_bar: Optional[_tqdm] = None
        self._test_progress_bar: Optional[_tqdm] = None
        self._predict_progress_bar: Optional[_tqdm] = None
        
        
        
    def __getstate__(Self) -> dict:
        return {
            key: value if not isinstance(value, _tqdm) else None
            for key, value in vars(self).items()
        }
        
    
    
    @property
    def train_progress_bar(self) -> _tqdm:
        if self._train_progress_bar is None:
            raise TypeError(f'The `{self.__class__.__name__}._train_progress_bar` reference has not been set yet')
        return self._train_progress_bar
    
    
    
    @train_progress_bar.setter
    def train_progress_bar(self, bar: _tqdm) -> None:
        self._train_progress_bar = bar
        
        
    @property
    def val_progress_bar(self) -> _tqdm:
        if self._val_progress_bar is None:
            raise TypeError(f'The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet.')
        return self._val_progress_bar
    
    
    
    @val_progress_bar.setter
    def val_progress_bar(self, bar: _tqdm) -> None:
        self._val_progress_bar = bar
        
        
        
    @property
    def test_progress_bar(self) -> _tqdm:
        if self._test_progress_bar is None:
            raise TypeError(f'The `{self.__class__.__name__}._test_progress_bar` reference has not been set yet.')
        return self._test_progress_bar
    
    
    
    @test_progress_bar.setter
    def test_progress_bar(self, bar: _tqdm) -> None:
        self._test_progress_bar = bar
        
        
    @property
    def predict_progress_bar(self) -> _tqdm:
        if self._predict_progress_bar is None:
            raise TypeError(f'The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet.')
        return self._predict_progress_bar
    
    
    
    @predict_progress_bar.setter
    def predict_progress_bar(self, bar: _tqdm) -> None:
        self._predict_progress_bar = bar
        
        
    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate
    
    
    @property
    def process_position(self) -> int:
        return self._process_position
    
    
    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled 
    
    
    def disable(self) -> None:
        self._enabled = True
        
        
    def init_sanity_tqdm(self) -> Tqdm:
        return Tqdm(
            desc= self.sanity_check_description,
            position= (2 * self.process_position),
            disable= self.is_disabled,
            leave= False,
            dynamic_ncols= True,
            file= sys.stdout,
            bar_format= self.bar_format
        )        
        
        
    def init_train_tqdm(self) -> Tqdm:
        return Tqdm(
            desc= self.train_description,
            position= (2 * self.process_position),
            disable= self.is_disabled,
            leave= True,
            dynamic_ncols= True,
            file= sys.stdout,
            smoothing= 0,
            bar_format= self.bar_format
        )


    def init_predict_tqdm(self) -> Tqdm:
        return Tqdm(
            desc= self.predict_description,
            position= (2 * self.process_position),
            disable= self.is_disabled,
            leave= True,
            dynamic_ncols= True,
            file= sys.stdout,
            smoothing= 0,
            bar_format= self.bar_format
        )


    def init_validation_tqdm(self) -> Tqdm:
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc= self.validation_description,
            position= (2 * self.process_position + has_main_bar),
            disable= self.is_disabled,
            leave= not has_main_bar,
            dynamic_ncols= True,
            file= sys.stdout,
            bar_format=self.bar_format
        )
        


    def init_test_tqdm(self) -> Tqdm:
        return Tqdm(
            desc= 'Testing',
            position= (2 * self.process_position),
            disable= self.is_disabled,
            leave= True,
            dynamic_ncols= True,
            file= sys.stdout,
            bar_format= self.bar_format
        )



    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()
        self.train_progress_bar = Tqdm(disable=True)  
        
    
    
    def on_sanity_check_end(self, *_: Any) -> None:
        self.val_progress_bar.close()
        self.train_progress_bar.close()
        

    def on_train_start(self, *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()


    def on_train_epoch_start(self, trainer: 'flame_modules.Trainer', *_: Any) -> None:
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")



    def on_train_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', batch: Any, batch_index: int) -> None:
        n: int = batch_index + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, flame_module))



    def on_train_epoch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, flame_module))



    def on_train_end(self, *_: Any) -> None:
        self.train_progress_bar.close()
        
        
        
    
    def on_validation_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        if not trainer.sanity_checking:
            self.val_progress_bar = self.init_validation_tqdm()



    def on_validation_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        if not self.has_dataloader_changed(dataloader_index):
            return

        self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
        self.val_progress_bar.initial = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_index}")



    def on_validation_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        n: int = batch_index + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)
    
        
    
    def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self.val_progress_bar.close()
        self.reset_dataloader_index_tracker()
        if self._train_progress_bar is not None and trainer.state.fn == 'fit':
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, flame_module))



    def on_test_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self.test_progress_bar = self.init_test_tqdm()




    def on_test_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        if not self.has_dataloader_changed(dataloader_index):
            return

        self.test_progress_bar.reset(convert_inf(self.total_test_batches_current_dataloader))
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(f"{self.test_description} DataLoader {dataloader_index}")
        
        

    def on_test_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: 'Step_Output', batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        n: int = batch_index + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)
            
    
    
    
    
    def on_test_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self.test_progress_bar.close()
        self.reset_dataloader_index_tracker()


    def on_predict_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self.predict_progress_bar = self.init_predict_tqdm()
        
        

    def on_predict_batch_start(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        if not self.has_dataloader_changed(dataloader_index):
            return

        self.predict_progress_bar.reset(convert_inf(self.total_predict_batches_current_dataloader))
        self.predict_progress_bar.initial = 0
        self.predict_progress_bar.set_description(f"{self.predict_description} DataLoader {dataloader_index}")



    def on_predict_batch_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', outputs: Any, batch: Any, batch_index: int, dataloader_index: int = 0) -> None:
        n: int = batch_index + 1
        if self._should_update(n, self.predict_progress_bar.total):
            _update_n(self.predict_progress_bar, n)


    

    def on_predict_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
        self.predict_progress_bar.close()
        self.reset_dataloader_index_tracker()



    def print(self, *args: Any, sep: Optional[str] = ' ', **kwargs: Any) -> None:
        active_progress_bar = None

        if self._train_progress_bar is not None and not self.train_progress_bar.disable:
            active_progress_bar = self.train_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)



    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)


    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            return 20
        return refresh_rate
    
    
    



def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x



def _update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()
        
        
        
#@: Driver code
if __name__.__contains__('__main__'):
    print('hemllo')