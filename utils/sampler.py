import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, is_adapt=False, max_time_wdw=None, adapt_batchsize=None) -> None:
        self.data_source = data_source
        self.is_adapt = is_adapt
        self.adapt_batchsize = adapt_batchsize
        self.max_time_wdw = max_time_wdw

    def __iter__(self) -> Iterator[int]:
        if not self.is_adapt:
            return iter(range(len(self.data_source)))
        else:
            if self.max_time_wdw is None:
                return iter(range(self.adapt_batchsize-1, len(self.data_source)))
            else:
                return iter(range(self.max_time_wdw, len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class BatchSampler(Sampler[List[int]]):

    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        # batch = []
        # for idx in self.sampler:
        #     batch.append(idx)
        #     if len(batch) == self.batch_size:
        #         yield batch
        #         batch = []
        # if len(batch) > 0 and not self.drop_first:
        #     yield batch
        batch = []
        for idx in self.sampler:
            idx += 1
            batch.extend([b for b in range(idx-self.batch_size, idx)])
            if len(batch) == self.batch_size:
                print('--> BATCHED', batch)
                yield batch
                batch = []

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size