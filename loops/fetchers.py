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
from typing import Iterator
from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException


# Class `_DataFetcher`:

# Within the "FlameModules" framework, the `_DataFetcher` serves as a specialized iterator engineered to fetch data from the 
# `CombinedLoader`. This class ensures seamless integration with the tracking and monitoring utilities of "FlameModules", 
# offering efficient mechanisms to manage exceptions and reset states during the data fetching process.

# Attributes:
#       - _combined_loader: An internal reference to the combined loader, a utility within "FlameModules" designed to 
#                           amalgamate multiple data loaders.
#       
#       - iterator: The primary iterator for data retrieval.
# 
#       - fetched: A counter that monitors the quantity of data batches retrieved.
# 
#       - done: A flag indicating the completion status of data fetching.
# 
#       - length: Specifies the total number of batches to extract from the combined loader.
# 
#       - _start_profiler and _stop_profiler: These lambdas act as hooks for the profiling mechanisms of "FlameModules".

# Methods:

#       - `combined_loader`: A property ensuring the safe acquisition of the `_combined_loader`. Triggers an exception 
#                            if the loader is uninitialized.
# 
#       - `setup`: Facilitates the configuration of the `_DataFetcher` with the provided combined loader.
# 
#       - `__iter__`: A canonical iterator method. It readies the iterator derived from the combined loader and reinstates 
#                     internal states.
# 
#       - `__next__`: Undertakes the task of fetching the succeeding data batch, simultaneously coordinating with the 
#                     "FlameModules" profiler and updating the fetched counter.
#       
#       - `reset`: Resets the fetched tally and ascertains the overall length of batches present in the combined loader.
#       
#       - `teardown`: Reverts the internal states to their original configuration and nullifies the iterator.


# Design Philosophy:

# 1. Seamless Integration: The `_DataFetcher` is meticulously crafted for "FlameModules", ensuring its flawless operation 
#                          within the package's ecosystem.
# 
# 2. Incorporation of Profiling Hooks: Its tight-knit integration with the profiling system of "FlameModules" empowers users 
#                                      to pinpoint and rectify potential performance issues related to data fetching.
# 
# 3. Emphasis on Robustness & Exception Management: With its built-in mechanisms, the class gracefully handles exceptions, 
#                                                   thus ensuring that potential disruptions during data extraction are 
#                                                   efficiently tackled.
# 
# 4. State Maintenance & Modularity**: Through utility methods like `reset` and `teardown`, the fetcher underscores the 
#                                      significance of modular state management, making it adaptable to diverse training 
#                                      scenarios within the "FlameModules" framework.

class _DataFetcher(Iterator): 
    def __init__(self) -> None:
        self._combined_loader: Optional[CombinedLoader] = None
        self.iterator: Optional[Iterator] = None
        self.fetched: Optional[int] = 0
        self.done: Optional[bool] = False
        self.length: Optional[int] = None
        self._start_profiler = lambda : None        #@: _profile_nothion -> def _profile_nothing() -> None: pass
        self._stop_profiler = lambda : None         #@: _profile_nothing -> def _profile_nothing() -> None: pass
        
    
    
    @property
    def combined_loader(self) -> CombinedLoader:
        if self._combined_loader is None:
            raise MisconfigurationException(
                f'`{self.__class__.__name__}` should have been `setup` with a `CombinedLoader`.'
            )
            
        return self._combined_loader
    
    
    
    def setup(self, combined_loader: CombinedLoader) -> None:
        self._combined_loader = combined_loader
        
    
    
    def __iter__(self) -> '_DataFetecher':
        self.iterator = iter(self.combined_loader)
        self.reset()
        return self
    
    
    
    def __next__(self) -> '_Iterator_return':
        assert self.iterator is not None
        self._start_profiler()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.done = True
            raise 
        finally:
            self._stop_profiler()
                
        self.fetched += 1
            
        if self.length is not None:
            self.done = self.fetched >= self.length
            
        return batch
        


    def reset(self) -> None:
        self.fetched = 0
        if self._combined_loader is not None:
            self.length = sized_len(self.combined_loader)
            self.done = self.length == 0
            
        
        
    def teardown(self) -> None:
        self.reset()
        if self._combined_loader is not None:
            self._combined_loader.reset()
        
        self.iterator = None
        
        

# Class `_PrefetchDataFetcher`:
# Deriving from `_DataFetcher`, the `_PrefetchDataFetcher` is a pivotal utility within the "FlameModules" framework, 
# engineered to expedite the data retrieval process by prefetching batches. By prefetching, it ensures that data is 
# readily available when demanded, optimizing data-handling efficiencies in computationally intensive scenarios.

# Attributes:
#       - prefetch_batches: Defines the number of batches to be prefetched. A minimum prefetching of one batch is 
#                           mandatory to ascertain whether a batch is the last one in the sequence.
#       
#       - batches: A dynamic list maintaining the prefetched batches, allowing for a FIFO (First-In, First-Out) 
#                  retrieval.

# Methods:
#       - `__init__`: An initializer that validates the prefetch batch count while establishing the superclass 
#                     configuration.
# 
#       - `__iter__`: Initiates the iteration process, delegating part of its operation to the superclass. It 
#                     manages the prefetching logic, ensuring that batches are fetched and stored in advance.
# 
#       - `__next__`: Orchestrates the batch retrieval process. It prioritizes the return of prefetched batches 
#                     and, in parallel, fetches subsequent batches to replenish the prefetch cache. If no prefetched
#                     batches are available, it defaults to the superclass's next batch retrieval.
#
#       - `reset`: Resets the prefetch batch list, ensuring that the prefetch cache is cleared, and also triggers 
#                  the superclass's reset method.

# Design Philosophy:
#       1. Acceleration Through Prefetching: The `_PrefetchDataFetcher` was architected with a singular goal: enhancing 
#                                            the batch retrieval rate. By prefetching batches, it diminishes potential 
#                                            data retrieval latencies during model training.
# 
#       2. Modularity through Inheritance: By extending `_DataFetcher`, the class absorbs all foundational functionalities
#                                          and only necessitates extensions that deal with prefetching, leading to a cleaner 
#                                          and more organized codebase.
#
#       3. Exception Handling & Robustness: Integrated exception handling mechanisms, like the ones managing `StopIteration`,
#                                           ensure that the class operates reliably even when data sources get exhausted.
#
#       4. Optimized Cache Management: The FIFO mechanism of managing prefetched batches ensures that memory usage remains in check 
#                                      and data retrieval remains seamless.

# This class encapsulates the best practices of data management, making it an invaluable addition to the "FlameModules" suite, 
# specifically for users who seek optimal performance during training sessions.

class _PrefetchDataFetcher(_DataFetcher):
    #@: This class is used to control the batch fetching flow
    # Args:
    #   *   prefetch_batches (int): Number of batches to pre-fetch. Pre-fetching atleast 1 batch is necessary
    #                               to properly track whether a batch is the last one. 
    #
    def __init__(self, prefetch_batches: Optional[int] = 1) -> None:
        super(_PrefetchDataFetcher, self).__init__()
        if prefetch_batches < 0:
            raise ValueError()
        
        self.prefetch_batches = prefetch_batches
        self.batches: list[Any] = []
        
    
    
    def __iter__(self) -> '_PrefetchDataFetcher':
        super(_PrefetchDataFetcher, self).__iter__()
        if self.length is not None:
            return self
        
        for _ in range(self.prefetch_batches):
            try:
                batch = super(_PrefetchDataFetcher, self).__next__()
                self.batches.append(batch)
                
            except StopIteration: break
            
        return self
    
    
    
    def __next__(self) -> '_Iterator_return':
        if self.batches:
            batch = self.batches.pop(0)
            try:
                self.batches.append(super(_PrefetchDataFetcher, self).__next__())
            except StopIteration: 
                self.done = not self.batches
            
        elif not self.done:
            batch = super(_PrefetchDataFetcher, self).__next__()
        
        else:
            raise StopIteration
        
        return batch
    
    
    
    def reset(self) -> None:
        super(_PrefetchDataFetcher, self).reset()
        self.batches: list[Any] = []
        


# Class _DataLoaderIterDataFetcher:
# Within the "FlameModules" ecosystem, _DataLoaderIterDataFetcher is a nuanced data fetcher, extending the capabilities of the _
# DataFetcher. It is formulated to offer direct access to the dataloader_iter, empowering developers using the FlameModule to 
# engineer custom prefetching strategies seamlessly.
#
# Attributes:
#       -  _batch: Stores the current batch data, dynamically updated with each iteration.
#       -  _batch_index: An integer counter to track the current batch's index.
#       -  _dataloader_index: Monitors the index of the data loader for scenarios involving multiple data loaders.
#
# Methods:
#       -  __init__: The constructor initializes the object, setting up the essential variables while also employing the base 
#                    class's initializer.
#
#       -  __iter__: Readies the object for iteration. It establishes an iterator wrapper using _DataFetcherWrapper, ensuring 
#                    that it's rightly encapsulated around the current object.
# 
#       -  __next__: Retrieves the next batch of data. If data fetching has been completed, it raises a StopIteration exception.
# 
#       -  reset: Resets the internal variables of the class, reverting the batch, batch index, and data loader index to their 
#                 original states. It also triggers the reset method from the superclass to maintain consistency.
#
# Design Philosophy:
#       1. Customization & Flexibility: By providing direct access to dataloader_iter, the class promotes customization, permitting 
#                                       developers to tailor the prefetching mechanism as per their application's needs.
# 
#       2. Layered Iteration Mechanism: With _DataFetcherWrapper encapsulating the current object, the class ensures a layered iteration, 
#                                       streamlining data fetching and offering advanced functionalities like tracking batch and data 
#                                       loader indices.
#
#       3. Consistent State Management: Integrated state management, from iteration initialization to resetting, ensures that the class's
#                                       internal state remains coherent throughout its lifecycle.
# 
#       4. Enhanced Developer Control: By exposing batch and data loader indices, the class bolsters developer control over data fetching 
#                                      processes, aiding in scenarios where intricate batch manipulations or data loader selections are 
#                                      imperative.
#
# The _DataLoaderIterDataFetcher serves as a testament to the "FlameModules" commitment to furnishing granular control to developers while 
# maintaining the simplicity of operation. The class stands as a beacon for those looking to integrate advanced data handling mechanisms 
# in their training workflows.

class _DataLoaderIterDataFetcher(_DataFetcher):
    #@: This class is used to return directly the `dataloader_iter` to the `FlameModule` training_step
    #@: to implement their own pre-fetching logic. This feature can be activated as follows:
    #
    #   class CustomNNModel(flame_modules.FlameModules | `nn.Module`):
    #       def __init__(self) -> None:
    #           ...
    #       
    #       def forward(self, x: torch.tensor) -> torch.tensor:
    #           ...
    #      
    #       def training_step(self, dataloader_iter: Iterator) -> None:
    #           fetch the batch from the dataloader and move that single `batch` to the right `torch.device`
    #           batch, batch_index, dataloader_index = dataloader_iter.__next__()
    #           batch = batch.to(self.device)
    #           ...
    #  
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(_DataLoaderIterDataFetcher, self).__init__(*args, **kwargs)
        self._batch: Any = None
        self._batch_index: int = 0
        self._dataloader_index: int = 0
        
    
    def __iter__(self) -> '_DataLoaderIterDataFetcher':
        super(_DataLoaderIterDataFetcher, self).__iter__()
        self.iterator_wrapper = iter(_DataFetcherWrapper(self))
        return self
    
    
    def __next__(self) -> Iterator['_DataFetcherWrapper']:
        if self.done:
            raise StopIteration
        
        return self.iterator_wrapper
    
    
    
    def reset(self) -> None:
        super(_DataLoaderIterDataFetcher, self).reset()
        self._batch = None
        self._batch_index = 0
        self._dataloader_index = 0
        
        
        
# Class _DataFetcherWrapper:
# An essential component in the "FlameModules" data handling ecosystem, the `_DataFetcherWrapper` acts as a specialized iterator 
# designed to wrap around `_DataLoaderIterDataFetcher`, bestowing additional utility and interface enhancements.
#
# Attributes:
#       -  data_fetcher: A reference to the associated `_DataLoaderIterDataFetcher` instance, the core object responsible for data retrieval.
#
# Properties:
#       -  done: A boolean flag indicating whether the data fetching process is completed.
#       -  fetched: An integer tracking the number of data batches fetched so far.
#       -  length: Represents the total length or count of data batches available for fetching.
#
# Methods:
#       -  __init__: Initializes the wrapper, setting up a reference to the given `_DataLoaderIterDataFetcher` instance.
#       -  __next__: Facilitates the iterative fetching of data batches. If data retrieval process is completed (`done` is `True`), 
#                    it raises a StopIteration exception. Otherwise, it retrieves the next batch of data, its index, and the data loader 
#                    index from the encapsulated data fetcher.
#
# Design Philosophy:
#       1. Seamless Integration: The wrapper is designed to integrate smoothly with the `_DataLoaderIterDataFetcher`, enhancing its 
#                                interface while maintaining the core functionality.
# 
#       2. Transparency & Proximity: By exposing properties like `done`, `fetched`, and `length`, the wrapper ensures users have easy 
#                                    access to essential data fetching metrics.
#
#       3. Efficiency in Iteration: The iterative mechanism (`__next__` method) is streamlined, ensuring a seamless data retrieval 
#                                   process. By storing crucial details within the encapsulated data fetcher, it ensures a consistent 
#                                   state across iterations.
#
#       4. Encapsulation & Extension: Acting as a layer over the `_DataLoaderIterDataFetcher`, the wrapper not only encapsulates the 
#                                     data fetcher but also extends its capabilities, thereby fortifying the data fetching mechanism.
#
# The `_DataFetcherWrapper` stands as a pivotal class in the "FlameModules" data handling suite. By reinforcing the data fetching process 
# and offering an enriched interface, it streamlines the developer experience and fortifies data processing workflows.

class _DataFetcherWrapper(Iterator):
    def __init__(self, data_fetcher: _DataLoaderIterDataFetcher) -> None:
        self.data_fetcher = data_fetcher
        
    
    @property
    def done(self) -> bool:
        return self.data_fetcher.done
    

    @property
    def fetched(self) -> int:
        return self.data_fetcher.fetched
    
    
    @property
    def length(self) -> Optional[int]:
        return self.data_fetcher.length
    

    def __next__(self) -> '_Iterator_return':
        fetcher = self.data_fetcher
        if fetcher.done:
            raise StopIteration
        
        batch, batch_index, dataloader_index = super(_DataLoaderIterDataFetcher, fetcher).__next__()
        
        fetcher._batch = batch
        fetcher._batch_index = batch_index
        fetcher._dataloader_index = dataloader_index
        return batch, batch_index, dataloader_index
    
        
#@: DRiver Code 
if __name__.__contains__('__main__'):
    print('hemllo')