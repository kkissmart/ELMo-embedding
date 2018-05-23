import itertools
import torch
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks".
    This is helpful in the case of recurrent models which use multiple gates
    applied to linear projections,
    which can be computed efficiently
    if they are concatenated together.
    They are separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """


    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise Exception("tensor dimensions must be divisible by their respective"
                            "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
                  for max_size, split in zip(sizes, split_sizes)]
        # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
            # A list of tuples containing the index to start at for this block
            # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
            # This is a tuple of slices corresponding to:
            # tensor[index: index + step_size, ...]. This is
            # required because we could have an arbitrary number
            # of dimensions. The actual slices we need are the
            # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                                 for start_index, step in index_and_step_tuples])
        tensor[block_slice] = torch.nn.init.orthogonal(tensor[block_slice].contiguous(), gain=gain)
