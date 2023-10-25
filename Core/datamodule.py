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

import inspect
from typing import IO, Any, Dict, Iterable, Optional, Union, cast

from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.hooks import DataHooks                                    # DONE : Datahooks {Understand}
from lightning.pytorch.core.mixins import HyperparametersMixin                        # DONE : mixins/HyperparametersMixin {Understand}
from lightning.pytorch.core.saving import _load_from_checkpoint
#from lightning.pytorch.utilities.model_helpers import _restricted_classmethod        # REVIEW
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

#@: First, do basic pytorch dataset 
#@: eq: my pose estimation dataset

# __note: str = r'''
#     In PyTorch, and in deep learning terminology in general, "sample" and "batch" are terms used to describe subsets of your dataset. 
#     Here's the differentiation:

#     *  Sample:
#         * A single element of your dataset.
#         * For example, in a dataset of images, a single image and its corresponding label or target would be one sample.


#     *  Batch:
#         * A group of samples.
#         * The number of samples in a batch is defined by the batch size, which is a hyperparameter you choose.

#     In PyTorch, when you use a DataLoader with a defined batch_size, it loads data in batches. Each batch contains multiple samples 
#     up to the specified batch size.
# '''


#@: BEGINS
#@: NOTE: https://github.com/rahul1-bot/Rahul_PyTorch_SpecialCode/blob/main/Pose%20Estimation/PoseEstimation_dataset.py
# class PoseEstimationDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_dir: str, annotation_file: str, 
#                                          transform_dict: Optional[dict[str, dict[str, dict[str, Callable[..., Any]]]]] = None) -> None:

        
#         self.image_dir: str = os.path.join(dataset_dir, 'images')
#         with open(os.path.join(dataset_dir, annotation_file), 'r') as f:
#             self.annotations: list[dict[str, Any]] = json.load(f)
        
#         self.transform_dict = transform_dict
        
        
        
                        
#     def __len__(self) -> int:
#         return len(self.annotations)
        
    
    
    
#     def __repr__(self) -> str:
#         return f'PoseEstimationDataset(num_samples= {len(self)}, transform_dict= {self.transform_dict})'
    
    
        
        
#     def __getitem__(self, index: int) -> dict[str, Union[torch.tensor, dict[str, torch.tensor]]]:
#         img_id: int = self.annotations[index]['image_id']
#         img_path: str = os.path.join(self.image_dir, f'{img_id}.jpg')
        
#         #@: getting all the dependent and independent_variables
#         image: PIL.Image = Image.open(img_path).convert('RGB')
#         keypoints: list[float] = self.annotations[index]['keypoints']
#         bounding_box: list[float] = self.annotations[index]['bounding_box']
#         pose_label: int = self.annotations[index]['pose_label']
#         metadata: dict[str, Any] = self.annotations[index]['metadata']
        
        
#         #@: transforms dependent and independent variable 
#         if self.transform_dict is not None:
#             if 'independent_variable' in self.transform_dict:
#                 for transform_func in self.transform_dict['independent_variable'].values():
#                     image: Any = transform_func(image)
                    
#             if 'dependent_variable' in self.transform_dict:
#                 if 'keypoints' in self.transform_dict['dependent_variable']:
#                     for transform_func in self.transform_dict['dependent_variable']['keypoints'].values():
#                         keypoints: list[float] = transform_func(keypoints)
            
#                 if 'bounding_box' in self.transform_dict['dependent_variable']:
#                     for transform_func in self.transform_dict['dependent_variable']['bounding_box'].values():
#                         bounding_box: list[float] = transform_func(bounding_box)
                        
#                 if 'pose_label' in self.transform_dict['dependent_variable']:
#                     for transform_func in self.transform_dict['dependent_variable']['pose_label'].values():
#                         pose_label: Any = transform_func(pose_label)
                
#                 if 'metadata' in self.transform_dict['dependent_variable']:
#                     for transform_func in self.transform_dict['dependent_variable']['metadata'].values():
#                         metadata: Any = transform_func(metadata)
                        
         
#         #@: converting dependent variable to tensors       
#         keypoints: torch.FloatTensor = torch.FloatTensor(keypoints)
#         bounding_box: torch.FloatTensor = torch.FloatTensor(bounding_box)
#         pose_label: torch.FloatTensor = torch.FloatTensor(pose_label)
#         metadata: torch.FloatTensor = torch.FloatTensor(metadata)
        
        
        
#         #@: returning a dict with independent_variable and dependent_variable
#         return {
#             'independent_variable': image,
#             'dependent_variable': {
#                 'keypoints': keypoints,
#                 'bounding_box': bounding_box,
#                 'pose_label': pose_label,
#                 'metadata': metadata
#             }
#         }



# #@: After DataModule 
# class PoseEstimationDataModule(DataModule):
#     def prepare_data(self) -> None:
#         ...
        
#     def setup(self, stage: str) -> None:
#         # make assignments here (val/train/test split)
#         # called on every process in DDP
#         dataset = PoseEstimationDataset(...)
#         self.train, self.val, self.test = data.random_split(
#             dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
#         )
    
    
#     def visualize_sample(self, index: int, dataset_type: Optional[str] = 'train') -> None:
#          ...

    
#     def train_dataloader(self) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(self.train)
    
    
#     def val_dataloader(self) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(self.val)
    
    
#     def test_dataloader(self) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(self.test)
    
    
#     def teardown(self) -> None:
#         # clean up state after the trainer stops, delete files...
#         # called on every process in DDP
#         ...
        
    
#     def state_dict(self) -> dict[str, Any]:
#         #@: Called when saving a checkpoint, implement to generate and save your datamodule class
#         ...


#     def load_state_dict(self, state_dict: dict[str, Any]) -> None:
#         #@: Called when loading a checkpoint, implement to reload your datamodule state from given datamodule `state_dict`
#         ...
        
        
#     def load_from_checkpoint(self, checkpoint_path, map_location, hparams_file= None, **kwargs):
#         ...
    

#@: DataModule
class DataModule(DataHooks, HyperparametersMixin):
    # A datamodule standardizes the training, val, test splits, data preparations and transforms. 
    # the main advantage is consistent data splits, data preparation and transforms across models
    def __init__(self) -> None:
        super().__init__()
        # Pointer to the trainer object
        self.trainer: Optional["pl.Trainer"] = None

    
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **datamodule_kwargs: Any,
    ) -> "LightningDataModule":
        
        def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        def train_dataloader() -> TRAIN_DATALOADERS:
            return apply_to_collection(train_dataset, Dataset, dataloader, shuffle=True)

        def val_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(val_dataset, Dataset, dataloader)

        def test_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(test_dataset, Dataset, dataloader)

        def predict_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(predict_dataset, Dataset, dataloader)

        candidate_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in accepted_params.values())
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard("self")
            special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}

        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader  # type: ignore[method-assign]
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader  # type: ignore[method-assign]
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader  # type: ignore[method-assign]
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader  # type: ignore[method-assign]
        return datamodule



    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        return {}



    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        ...



    #@_restricted_classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        **kwargs: Any,
    ) -> Self:
        # Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint it stores the
        # arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.

        # Any arguments specified through \*\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.

        # Args:
        #     checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
        #     map_location:
        #         If your checkpoint saved a GPU model and you now load on CPUs
        #         or a different number of GPUs, use this to map to the new setup.
        #         The behaviour is the same as in :func:`torch.load`.
        #     hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
        #         as in this example::

        #             dataloader:
        #                 batch_size: 32

        #         You most likely won't need this since Lightning will always save the hyperparameters
        #         to the checkpoint.
        #         However, if your checkpoint weights don't have the hyperparameters saved,
        #         use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
        #         These will be converted into a :class:`~dict` and passed into your
        #         :class:`LightningDataModule` for use.

        #         If your datamodule's ``hparams`` argument is :class:`~argparse.Namespace`
        #         and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat
        #         ``hparams`` as :class:`~dict`.
        #     \**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved
        #         hyperparameter values.

        # Return:
        #     :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).

        # Note:
        #     ``load_from_checkpoint`` is a **class** method. You must use your :class:`LightningDataModule`
        #     **class** to call it instead of the :class:`LightningDataModule` instance, or a
        #     ``TypeError`` will be raised.

        # Example::

        #     # load weights without mapping ...
        #     datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

        #     # or load weights and hyperparameters from separate files.
        #     datamodule = MyLightningDataModule.load_from_checkpoint(
        #         'path/to/checkpoint.ckpt',
        #         hparams_file='/path/to/hparams_file.yaml'
        #     )

        #     # override some of the params with new values
        #     datamodule = MyLightningDataModule.load_from_checkpoint(
        #         PATH,
        #         batch_size=32,
        #         num_workers=10,
        #     )

        
        loaded = _load_from_checkpoint(
            cls,  
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=None,
            **kwargs,
        )
        return cast(Self, loaded)



