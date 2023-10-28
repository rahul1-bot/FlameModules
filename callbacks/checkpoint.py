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
from callback import Callback

# Checkpoints in AI:
#
# Checkpoints are a crucial mechanism in the world of deep learning and AI training processes. They 
# represent saved states of a model at particular intervals or conditions during training. This saved 
# state can include the model weights, optimizer state, epoch number, and other related training metadata.
#
# Why Do We Need Checkpoints?
#
#     1. Recovery from Failures: Training deep learning models can be time-consuming, often extending to 
#                                several hours or even days. In the event of a system malfunction or crash, 
#                                without the use of checkpoints, the entire training would have to be 
#                                restarted from the beginning.
# 
#     2. Stopping and Resuming Training: There might be instances when a training session needs to be 
#                                        interrupted, perhaps due to system maintenance, or a need to adjust 
#                                        hyperparameters. Checkpoints enable the resumption of training from 
#                                        the last saved state, rather than starting afresh.
# 
#     3. Evaluation & Deployment: As the training of models progresses, there is often a need to evaluate 
#                                 the models on validation datasets to assess their performance. Checkpoints 
#                                 give access to models at different stages of training, facilitating periodic 
#                                 evaluation.
# 
#     4. Preventing Overfitting: Sometimes, as training progresses, models might start to overfit. 
#                                With checkpoints, one can revert to a model version from an earlier stage, 
#                                before overfitting began.
#
# Types of Checkpoints in AI:
#
#     1. Epoch-based Checkpoints: These are the most common type where the model state is saved after 
#                                 every epoch (or a set number of epochs).
# 
#     2. Performance-based Checkpoints: The model is saved whenever there's an improvement in its 
#                                        performance, for instance, when validation accuracy increases.
# 
#     3. Iteration-based Checkpoints: Instead of waiting for an epoch to finish, the model is saved after 
#                                     a specified number of batches or iterations.
# 
#     4. Conditional Checkpoints: These are saved based on custom conditions set by the user, which could 
#                                  be a combination of various metrics or other custom criteria.
#
# Utilizing checkpoints in training procedures ensures a more robust and flexible training process, 
# safeguarding against potential data loss and enabling various strategies for model evaluation and deployment.

class Checkpoint(Callback):
    #@: This is a base class for model checkpointing. 
    #@: Here users may want to subclass it in case of writing a custom :class: `flame_modules.callbacks.Checkpoint`
    #@: so that the `flame_module.Trainer` recognizes the custom class as a checkpointing callback
    ...


# :: NOTE :: Example ::
# import os

# class Checkpoint(Callback):
#     def __init__(self, save_path: str, monitor: str = 'val_loss', mode: str = 'min', max_save: int = 5, save_freq: int = 1) -> None:
#         super(Checkpoint, self).__init__()
        
#         self.save_path = save_path
#         self.monitor = monitor
#         self.mode = mode
#         self.max_save = max_save
#         self.save_freq = save_freq
#         self.best_score = float('inf') if mode == 'min' else float('-inf')
#         self.checkpoints_saved: list[Any] = []


#     def should_save_checkpoint(self, current_score: float) -> bool:
#         if self.mode == 'min':
#             return current_score < self.best_score
#         else:
#             return current_score > self.best_score



#     def on_validation_end(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule') -> None:
#         #@: Save model checkpoint after validation.
#         epoch = trainer.current_epoch  # Assuming the trainer object has current_epoch attribute
#         current_score = trainer.logged_metrics[self.monitor]  # Assuming the trainer object logs metrics
        
#         if epoch % self.save_freq == 0 and self.should_save_checkpoint(current_score):
#             checkpoint_path = f"{self.save_path}/model_checkpoint_epoch_{epoch}.pth"
#             self.save_checkpoint(flame_module, checkpoint_path)
#             self.best_score = current_score
#             self.checkpoints_saved.append(checkpoint_path)
            
#             # Ensure we don't exceed the max number of checkpoints to save
#             if len(self.checkpoints_saved) > self.max_save:
#                 os.remove(self.checkpoints_saved.pop(0))



#     def on_save_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
#         #@: Save model, optimizer, scheduler, and other states.
#         checkpoint['model_state_dict'] = flame_module.state_dict()
#         #@: Add other states if necessary, e.g., optimizer, scheduler



#     def on_load_checkpoint(self, trainer: 'flame_modules.Trainer', flame_module: 'flame_modules.FlameModule', checkpoint: dict[str, Any]) -> None:
#         #@: Load model and other states from the checkpoint.
#         flame_module.load_state_dict(checkpoint['model_state_dict'])
#         # Load other states if necessary, e.g., optimizer, scheduler

    
    