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
import logging, os
from argparse import Namespace
from typing_extensions import override

from lightning.fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning.fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.types import _PATH

# from FlameModules.Core.saving import save_hparams_to_ymal :: NOTE :: Not Done
from lightning.pytorch.core.saving import save_hparams_to_yaml
from FlameModules.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


log = logging.getLogger(__name__)



#@: :: NOTE :: ToDo :: Add Code Docx
class ExperimentWriter(_FabricExperimentWriter):
    #@: Experiment writer for CSVLogger.    
    name_hparams_file: str = 'hparams.yaml'
    
    
    def __init__(self, log_dir: str) -> None:
        super(ExperimentWriter, self).__init__(log_dir= log_dir)
        self.hparams: dict[str, Any] = {}
        
        
        
    def log_hparams(self, params: dict[str, Any]) -> None:
        self.hparams.update(params)
        
        
        
    @override
    def save(self) -> None:
        #@: Save recorded hparams and metrics into files
        hparams_file = os.path.join(self.log_dir, self.name_hparams_file)
        save_hparams_to_yaml(hparams_file, self.hparams)
        return super().save()
    
    
    
    
#@: :: NOTE :: ToDo :: Add Code Docx
class CSVLogger(Logger, FabricCSVLogger):
    #@: Log to local file system in ymal and CSV format.
    # Logs are saved to `os.path.join(save_dir, name, version)`.
    #
    # Code Doc:
    #       import FlameModules as flame_modules
    #       from flame_modules.trainer.trainer import Trainer
    #       from flame_modules.loggers.csv_logs import CSVLogger
    #
    #       if __name__.__contains__('__main__'):
    #           logger = CSVLogger('logs', name= 'my_exp_name')
    #           trainer = Trainer(logger= logger)
    #
    
    logger_join_char: str = '-'
    
    def __init__(self, save_dir: _PATH, name: Optional[str] = 'flame_logs', version: Optional[Union[int, str]] = None, prefix: Optional[str] = '', 
                                                                                                                       flush_logs_every_n_steps: Optional[int] = 100) -> None:
        super(CSVLogger, self).__init__(root_dir= save_dir, name= name, version= version, prefix= prefix, flush_logs_every_n_steps= flush_logs_every_n_steps)
        self._save_dir = os.fspath((save_dir))
        
    
    
    @property
    @override
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f'version_{self.version}'
        return os.path.join(self.root_dir, version)
    
    
    
    
    @property
    @override
    def save_dir(self) -> str:
        return self._save_dir
    
    
    
    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self.experiment.log_hparams(params)
        
        
        
    @property
    @override
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        if self._experiment is not None:
            return self._experiment
        
        self._fs.makedirs(self.root, exist_ok= True)
        self._experiment = ExperimentWriter(log_dir= self.log_dir)
        return self._experiment
    
    
    
    
