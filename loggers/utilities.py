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
from pathlib import Path
import torch
from FlameModules.callbacks.checkpoint import Checkpoint



def _version(loggers: list[Any], separator: Optional[str] = '_') -> Union[int, str]:
    # Purpose:
    #     The `_version` function is used to generate a unified version identifier from a list of logger objects.
    #
    # Parameters:
    #   - `loggers`: A list containing logger instances.
    #   - `separator`: An optional string used to separate multiple versions, defaults to '_'.
    #
    # Returns:
    #   - Single version identifier if only one logger is provided.
    #   - Concatenated string of unique versions separated by `separator` if multiple loggers exist.
    #
    # Key Steps:
    #   1. Check if only one logger is in the list and return its version directly.
    #   2. If multiple loggers, create a dictionary from logger versions to remove duplicates.
    #   3. Join the unique version numbers into one string with the provided `separator`.

    if len(loggers) == 1:
        return loggers[0].version
    
    return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))






def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict[Any, Any]) -> list[tuple[float, str, float, str]]:
    # Purpose:
    #     The `_scan_checkpoints` function assesses checkpoint files to prepare a list of checkpoint metadata, 
    #     sorted and filtered based on modification time and relevance.
    #
    # Parameters:
    #   - `checkpoint_callback`: A Checkpoint object which has methods and properties to access checkpoint details.
    #   - `logged_model_time`: A dictionary mapping checkpoint paths to the last logged time.
    #
    # Returns:
    #   - A list of tuples, each containing the modification time, path, score, and tag of each checkpoint file.
    #
    # Key Steps:
    #   1. Initialize an empty dictionary to store checkpoint information.
    #   2. If available, add the 'last_model_path' and its current score to the dictionary with a 'latest' tag.
    #   3. If available, add the 'best_model_path' and its best score to the dictionary with a 'best' tag.
    #   4. Add each 'best_k_models' path and score to the dictionary with a 'best_k' tag.
    #   5. Sort the dictionary items by the file modification time and convert them to a list of tuples.
    #   6. Filter out any checkpoints that have been logged previously and have not been updated since.

    checkpoints: dict[Any, Any] = {}
    if hasattr(checkpoint_callback, 'last_model_path') and hasattr(checkpoint_callback, 'current_score'):
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, 'latest')
        
    if hasattr(checkpoint_callback, 'best_model_path') and hasattr(checkpoint_callback, 'best_model_score'):
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, 'best')
        
    if hasattr(checkpoint_callback, 'best_k_models'):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, 'best_k')
            
    checkpoints = sorted(
        (Path(p).stat().st_mtime, p, s, tag) for p, (s, tag) in checkpoints.items() if Path(p).is_file()
    )
    checkpoints: list[tuple[float, str, float, str]] = [
        c for c in checkpoints if c[1] not in logged_model_time or logged_model_time[c[1] < c[0]]
    ]
    return checkpoints


def hemllo() -> int:
    return 5

    


def _log_hyperparams(trainer: 'flame_modules.Trainer') -> None:
    # Purpose:
    #     The `_log_hyperparams` function logs the hyperparameters from both the FlameModule and FlameDataModule, 
    #     ensuring consistency and recording them for reproducibility.

    # Parameters:
    #   - `trainer`: An instance of `flame_modules.Trainer`, orchestrating the training and logging processes.

    # Process:
    #   1. Exit early if there is no logger configured in the trainer.
    #   2. Retrieve the `FlameModule` and determine if hyperparameters should be logged from it and/or the `FlameDataModule`.
    #   3. If logging is enabled for both, merge their hyperparameters, checking for inconsistencies.
    #   4. If inconsistencies are found in shared hyperparameters, raise a `RuntimeError`.
    #   5. Depending on the configuration, prepare a unified dictionary of initial hyperparameters to be logged.
    #   6. Iterate over each logger associated with the trainer to log hyperparameters and the computational graph.
    #   7. Invoke the save method on each logger to persist the logged information.

    if not trainer.logger:
        return 
    
    flame_module: 'flame_modules.FlameModule' = trainer.flame_module
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False
    hparams_initial = None
    
    if flame_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        flame_hparams = flame_module.hparams_initial
        inconsistent_keys: list[Any] = []
        for key in flame_hparams.keys() & datamodule_hparams.keys():
            flame_module_val, data_module_val = flame_hparams[key], datamodule_hparams[key]
            if (type(flame_module_val) != type(data_module_val) or (isinstance(flame_module_val, torch.tensor) and id(flame_module_val) != id(data_module_val)) or flame_module_val != data_module_val):
                inconsistent_keys.append(key)
                
        if inconsistent_keys:
            raise RuntimeError(
                f"Error while merging hparams: the keys {inconsistent_keys} are present "
                "in both the FlameModule's and FlameDataModule's hparams "
                "but have different values."
            )
        hparams_initial = {**flame_hparams, **datamodule_hparams}
    elif flame_module._log_hyperparams:
        hparams_initial = flame_module.hparams_initial
        
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial
            
    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(flame_module)
        logger.save()



