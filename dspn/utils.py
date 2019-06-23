import scipy.optimize
import torch
import torch.nn.functional as F


def hungarian_loss(predictions, targets, thread_pool):
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets, reduction="none").mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()
    indices = thread_pool.map(hungarian_loss_per_sample, squared_error_np)
    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(squared_error, indices)
    ]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss


def hungarian_loss_per_sample(sample_np):
    return scipy.optimize.linear_sum_assignment(sample_np)


def chamfer_loss(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets, reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)


def scatter_masked(tensor, mask, binned=False, threshold=None):
    s = tensor[0].detach().cpu()
    mask = mask[0].detach().clamp(min=0, max=1).cpu()
    if binned:
        s = s * 128
        s = s.view(-1, s.size(-1))
        mask = mask.view(-1)
    if threshold is not None:
        keep = mask.view(-1) > threshold
        s = s[:, keep]
        mask = mask[keep]
    return s, mask


def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b
