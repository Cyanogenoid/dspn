import torch
import torch.nn as nn
import torch.nn.functional as F


class FSPool(nn.Module):
    """
        Featurewise sort pooling. From:

        FSPool: Learning Set Representations with Featurewise Sort Pooling.
        Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
        https://arxiv.org/abs/1906.02795
        https://github.com/Cyanogenoid/fspool
    """

    def __init__(self, in_channels, n_pieces, relaxed=False):
        """
        in_channels: Number of channels in input
        n_pieces: Number of pieces in piecewise linear
        relaxed: Use sorting networks relaxation instead of traditional sorting
        """
        super().__init__()
        self.n_pieces = n_pieces
        self.weight = nn.Parameter(torch.zeros(in_channels, n_pieces + 1))
        self.relaxed = relaxed

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, n=None):
        """ FSPool

        x: FloatTensor of shape (batch_size, in_channels, set size).
        This should contain the features of the elements in the set.
        Variable set sizes should be padded to the maximum set size in the batch with 0s.

        n: LongTensor of shape (batch_size).
        This tensor contains the sizes of each set in the batch.
        If not specified, assumes that every set has the same size of x.size(2).
        Note that n.max() should never be greater than x.size(2), i.e. the specified set size in the
        n tensor must not be greater than the number of elements stored in the x tensor.

        Returns: pooled input x, used permutation matrix perm
        """
        assert x.size(1) == self.weight.size(
            0
        ), "incorrect number of input channels in weight"
        # can call withtout length tensor, uses same length for all sets in the batch
        if n is None:
            n = x.new(x.size(0)).fill_(x.size(2)).long()
        # create tensor of ratios $r$
        sizes, mask = fill_sizes(n, x)
        mask = mask.expand_as(x)

        # turn continuous into concrete weights
        weight = self.determine_weight(sizes)

        # make sure that fill value isn't affecting sort result
        # sort is descending, so put unreasonably low value in places to be masked away
        x = x + (1 - mask).float() * -99999
        if self.relaxed:
            x, perm = cont_sort(x, temp=self.relaxed)
        else:
            x, perm = x.sort(dim=2, descending=True)

        x = (x * weight * mask.float()).sum(dim=2)
        return x, perm

    def forward_transpose(self, x, perm, n=None):
        """ FSUnpool 

        x: FloatTensor of shape (batch_size, in_channels)
        perm: Permutation matrix returned by forward function.
        n: LongTensor fo shape (batch_size)
        """
        if n is None:
            n = x.new(x.size(0)).fill_(perm.size(2)).long()
        sizes, mask = fill_sizes(n)
        mask = mask.expand(mask.size(0), x.size(1), mask.size(2))

        weight = self.determine_weight(sizes)

        x = x.unsqueeze(2) * weight * mask.float()

        if self.relaxed:
            x, _ = cont_sort(x, perm)
        else:
            x = x.scatter(2, perm, x)
        return x, mask

    def determine_weight(self, sizes):
        """
            Piecewise linear function. Evaluates f at the ratios in sizes.
            This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
        """
        # share same sequence length within each sample, so copy weighht across batch dim
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(sizes.size(0), weight.size(1), weight.size(2))

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * sizes
        index = index.unsqueeze(1)
        index = index.expand(index.size(0), weight.size(1), index.size(2))

        # points in the weight vector to the left and right
        idx = index.long()
        frac = index.frac()
        left = weight.gather(2, idx)
        right = weight.gather(2, (idx + 1).clamp(max=self.n_pieces))

        # interpolate between left and right point
        return (1 - frac) * left + frac * right


def fill_sizes(sizes, x=None):
    """
        sizes is a LongTensor of size [batch_size], containing the set sizes.
        Each set size n is turned into [0/(n-1), 1/(n-1), ..., (n-2)/(n-1), 1, 0, 0, ..., 0, 0].
        These are the ratios r at which f is evaluated at.
        The 0s at the end are there for padding to the largest n in the batch.
        If the input set x is passed in, it guarantees that the mask is the correct size even when sizes.max()
        is less than x.size(), which can be a case if there is at least one padding element in each set in the batch.
    """
    if x is not None:
        max_size = x.size(2)
    else:
        max_size = sizes.max()
    size_tensor = sizes.new(sizes.size(0), max_size).float().fill_(-1)

    size_tensor = torch.arange(end=max_size, device=sizes.device, dtype=torch.float32)
    size_tensor = size_tensor.unsqueeze(0) / (sizes.float() - 1).clamp(min=1).unsqueeze(
        1
    )

    mask = size_tensor <= 1
    mask = mask.unsqueeze(1)

    return size_tensor.clamp(max=1), mask.float()


def deterministic_sort(s, tau):
    """
    "Stochastic Optimization of Sorting Networks via Continuous Relaxations" https://openreview.net/forum?id=H1eSS3CcKX

    Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon

    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar.
    """
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=s.device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.transpose(0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).type(torch.float32)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def cont_sort(x, perm=None, temp=1):
    """ Helper function that calls deterministic_sort with the right shape.
    Since it assumes a shape of (batch_size, n, 1) while the input x is of shape (batch_size, channels, n),
    we can get this to the right shape by merging the first two dimensions.
    If an existing perm is passed in, we compute the "inverse" (transpose of perm) and just use that to unsort x.
    """
    original_size = x.size()
    x = x.view(-1, x.size(2), 1)
    if perm is None:
        perm = deterministic_sort(x, temp)
    else:
        perm = perm.transpose(1, 2)
    x = perm.matmul(x)
    x = x.view(original_size)
    return x, perm


if __name__ == "__main__":
    pool = FSort(2, 1)
    x = torch.arange(0, 2 * 3 * 4).view(3, 2, 4).float()
    print("x", x)
    y, perm = pool(x, torch.LongTensor([2, 3, 4]))
    print("perm")
    print(perm)
    print("result")
    print(y)
