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
from progress import _BaseProgress


# NOTE : Class `_Loop`: NOTE

# This class manages the state and progress of loop constructs typically seen in model training processes. Its design aims 
# to facilitate interruption-resistant training sessions by saving and restoring states, ensuring continuity and robustness.

# Attributes:
#     trainer: A reference to the associated training module or handler.
#     _restarting: A boolean flag that indicates whether the loop is in a restarting phase.


# Methods:
#     - `restarting`: Property method to safely get and set the `_restarting` flag. The setter ensures propagation 
#                     of the restarting state to child loops, ensuring consistent state management across nested loops.

#     - `on_save_checkpoint`: A hook meant for saving the current state of the loop. This method should be expanded upon 
#                             by subclasses to customize the checkpointing process.

#     - `on_load_checkpoint`: A hook meant for restoring the loop's state from a checkpoint. Like its saving counterpart, 
#                             it's designed for subclasses to define specific restoration processes.

#     - `state_dict`: Returns a dictionary capturing the current state of the loop, including any progress trackers and 
#                     nested loop structures. The prefixing system distinguishes between states of nested loops.

#     - `load_state_dict`: Accepts a state dictionary to restore the state of the loop and its children. It ensures a 
#                          recursive restoration for nested loops. After state restoration, the `restarting` flag is set 
#                          to True.

#     - `_load_from_state_dict`: An internal method to restore the specific state of this loop (excluding its children) 
#                                from the provided state dictionary.


# Design Philosophy:

# 1. State Persistence: The architecture places an emphasis on capturing and restoring states, catering to the requirements 
#                       of extensive deep learning workflows where training sessions could be interrupted and need resumption.

# 2. Hierarchical Management: The structure acknowledges the nested nature of training loops (e.g., epoch loop containing 
#                             batch loop) and handles their states seamlessly.

# 3. Modular & Extendable: Hook methods (`on_save_checkpoint` and `on_load_checkpoint`) allow subclasses to define custom 
#                          state management logic, promoting modularity.

# 4. Safety & Consistency: The class ensures consistent state across its hierarchy, especially when states are modified, 
#                          enhancing reliability during training processes.


class _Loop:
    #  Represents a generic loop structure, e.g., a training or evaluation loop, in the training framework.
    
    # Attributes:
    #     _restarting (bool): Internal flag indicating if the loop needs a restart.
    #     trainer: Reference to the associated trainer instance.

    # Methods:
    #     - __init__(trainer: 'flame_modules.Trainer'): Initializes the loop with a reference to a trainer.
    #     - on_save_checkpoint() -> dict[Any, Any]: Captures the loop's state when saving a checkpoint.
    #     - on_load_checkpoint(state_dict: dict[Any, Any]): Populates loop's state from a checkpoint.
    #     - state_dict(destination: Optional[dict[Any, Any]] = None, prefix: Optional[str] = '') -> dict[Any, Any]: 
    #         Captures the current state of the loop and its children (loops and progress trackers).
    #     - load_state_dict(state_dict: dict[Any, Any], prefix: Optional[str] = ''): Sets the loop's state and its children 
    #         from the provided state dictionary.
    #     - _load_from_state_dict(state_dict: dict[Any, Any], prefix: str): Internal method to assist in populating the loop's state.

    # Properties:
    #     - restarting: Indicates whether the loop is in a restarting state.        
    def __init__(self, trainer: 'flame_modules.Trainer') -> None:
        self._restarting: bool = False
        self.trainer = trainer
        
        
    @property
    def restarting(self) -> bool:
        #@: Whether the state of this loop was reloaded and it needs to restart 
        return self._restarting
    
    
    @restarting.setter
    def restarting(self, restarting: bool) -> None:
        #@: Connents this loop's restarting value and its children
        self._restarting = restarting
        for loop in vars(self).values():
            if isinstance(loop, _Loop):
                loop.restarting = restarting
                
        
                
    def on_save_checkpoint(self) -> dict[Any, Any]:
        #@: Called when saving a model checkpoint, use to persist loop state.
        return {}
    
    
    def on_load_checkpoint(self, state_dict: dict[Any, Any]) -> None:
        #@: Called when loading a model checkpoint, use to reload loop state. 
        ...
        
    
    def state_dict(self, destination: Optional[Union[dict[Any, Any]], NoneType] = None, prefix: Optional[str] = '') -> dict[Any, Any]:
        #@: The state dict is determined by the state and the progress of this loop and its childrens. 
        # Args:
        #   * destination (dict[Any, Any] | NoneType): An existing dictionary to update with this loop's state. 
        #                                              By default a new dictionary is returned. 
        #
        #   * prefix (str): A prefix for each key in the state dictionary. 
        #
        if destination is None:
            destination: dict[Any, Any] = {}
        
        destination[prefix + 'state_dict'] = self.on_save_checkpoint()
        
        for key, val in self.__dict__.items():
            key: str = prefix + key
            if isinstance(val, _BaseProgress):
                destination[key] = val.state_dict()
            
            elif isinstance(val, _Loop):
                val.state_dict(destination, key + '.')
            
        return destination
    
    
    
    
    def load_state_dict(self, state_dict: dict[Any, Any], prefix: Optional[str] = '') -> None:
        #@: Loads the state of this loop and all its childrens 
        self._load_from_state_dict(state_dict.copy(), prefix)
        
        for key, val in self.__dict__.items():
            if isinstance(val, _Loop):
                val.load_state_dict(state_dict.copy(), prefix + key + '.')
        
        self.restarting = True 
        
    
    
    
    def _load_from_state_dict(self, state_dict: dict[Any, Any], prefix: str) -> None:
        for key, val in self.__dict__.items():
            key: str = prefix + key
            
            if key not in state_dict:
                continue
            
            if isinstance(val, _BaseProgress):
                val.load_state_dict(state_dict[key])
                
        if prefix + 'state_dict' in state_dict:
            self.on_load_checkpoint(state_dict[prefix + 'state_dict'])
            
            
    
            

#@: Driver Code
if __name__.__contains__('__main__'):
    print('hemllo')
    
