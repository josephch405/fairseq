# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset, plasma_utils

from tqdm import tqdm

class CurriculumDataset(BaseWrapperDataset):
    """Randomly samples from a given dataset at each epoch.

    Sampling is done with or without replacement, depending on the "replace"
    parameter.

    Optionally, the epoch size can be rescaled. This is potentially desirable
    to increase per-epoch coverage of the base dataset (since sampling with
    replacement means that many items in the dataset will be left out). In the
    case of sampling without replacement, size_ratio should be strictly less
    than 1.

    Args:
        dataset (~torch.utils.data.Dataset): dataset on which to sample.
        weights (List[float]): list of probability weights
            (default: None, which corresponds to uniform sampling).
        replace (bool): sampling mode; True for "with replacement", or False
            for "without replacement" (default: True)
        size_ratio (float): the ratio to subsample to; must be positive
            (default: 1.0).
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 0).
        epoch (int): starting epoch number (default: 1).
    """

    def __init__(
        self,
        dataset,
        replace=True,
        size_ratio=1,
        batch_by_size=True,
        seed=0,
        epoch=1,
        bias=0.1,
        curriculum_length=10 # TODO: try to refactor out of epochs based length
    ):
        super().__init__(dataset)

        # if weights is None:
        #     self.weights = None
        # else:
        #     assert len(weights) == len(dataset)
        #     weights_arr = np.array(weights, dtype=np.float64)
        #     weights_arr /= weights_arr.sum()
        #     self.weights = plasma_utils.PlasmaArray(weights_arr)

        self.replace = replace

        assert size_ratio > 0.0
        if not self.replace:
            assert size_ratio < 1.0
        self.size_ratio = float(size_ratio)
        self.actual_size = np.ceil(len(dataset) * self.size_ratio).astype(int)

        self.bias = bias
        # TODO: replace with pacing module
        self.slope = 1 / (self.actual_size * curriculum_length)

        self.batch_by_size = batch_by_size
        self.seed = seed

        self._cur_epoch = None
        self._cur_indices = None

        self.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[self._cur_indices.array[index]]

    def collater(self, samples):
        return self.dataset.collater(samples)

    def __len__(self):
        return self.actual_size

    @property
    def sizes(self):
        # We sort by source sizes for now
        if isinstance(self.dataset.src_sizes, list):
            return [s[self._cur_indices.array] for s in self.dataset.src_sizes]
        return self.dataset.src_sizes[self._cur_indices.array]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self._cur_indices.array[index])

    def size(self, index):
        return self.dataset.size(self._cur_indices.array[index])

    def ordered_indices(self):
        if self.batch_by_size:
            order = [
                np.arange(len(self)),
                self.sizes,
            ]  # No need to handle `self.shuffle == True`
            return np.lexsort(order)
        else:
            return np.arange(len(self))

    def prefetch(self, indices):
        self.dataset.prefetch(self._cur_indices.array[indices])

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

        if epoch == self._cur_epoch:
            return

        self._cur_epoch = epoch

        # Generate a weighted sample of indices as a function of the
        # random seed and the current epoch.

        rng = np.random.RandomState(
            [
                42,  # magic number
                self.seed % (2 ** 32),  # global seed
                self._cur_epoch,  # epoch index
            ]
        )

        sample_batch = 100
        chosen_indices = np.array([], dtype=int)
        i = 0
        weights = np.zeros(len(self.dataset), dtype=np.float64)
        while i < self.actual_size:
            t = (epoch - 1) * self.actual_size + i
            competency = min(1, self.bias + t * self.slope)
            max_difficulty = np.percentile(self.dataset.src_sizes, competency * 100)
            # filter based on length of sources
            # TODO: incorporate other sampling scores
            passes_filter = (self.dataset.src_sizes < max_difficulty)
            weights[passes_filter] = 1

            weights /= weights.sum()
            chosen_indices = np.append(chosen_indices, rng.choice(
                len(self.dataset),
                min(sample_batch, self.actual_size - t),
                p=weights,
            ))
            i += sample_batch

        self._cur_indices = plasma_utils.PlasmaArray(
            chosen_indices
        )
