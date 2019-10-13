import torch
import torch.nn as nn
import torch.nn.functional as F


class DSPN(nn.Module):
    """ Deep Set Prediction Networks
    Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
    NeurIPS 2019
    https://arxiv.org/abs/1906.06565
    """

    def __init__(self, encoder, set_channels, max_set_size, channels, iters, lr):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.

        channels: Number of channels of the set to predict.

        max_set_size: Maximum size of the set.

        iter: Number of iterations to run the DSPN algorithm for.

        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr
        self.channels = channels

        self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
        self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))

    def forward(self, target_repr):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.

        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        # copy same initial set over batch
        current_set = self.starting_set.expand(
            target_repr.size(0), *self.starting_set.size()[1:]
        )
        current_mask = self.starting_mask.expand(
            target_repr.size(0), self.starting_mask.size()[1]
        )
        # make sure mask is valid
        current_mask = current_mask.clamp(min=0, max=1)

        # info used for loss computation
        intermediate_sets = [current_set]
        intermediate_masks = [current_mask]
        # info used for debugging
        repr_losses = []
        grad_norms = []

        # optimise repr_loss for fixed number of steps
        for i in range(self.iters):
            # regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
            with torch.enable_grad():
                if not self.training:
                    current_set.requires_grad = True
                    current_mask.requires_grad = True

                # compute representation of current set
                predicted_repr = self.encoder(current_set, current_mask)
                # how well does the representation matches the target
                repr_loss = F.smooth_l1_loss(
                    predicted_repr, target_repr, reduction="mean"
                )
                # change to make to set and masks to improve the representation
                set_grad, mask_grad = torch.autograd.grad(
                    inputs=[current_set, current_mask],
                    outputs=repr_loss,
                    only_inputs=True,
                    create_graph=True,
                )
            # update set with gradient descent
            current_set = current_set - self.lr * set_grad
            current_mask = current_mask - self.lr * mask_grad
            current_mask = current_mask.clamp(min=0, max=1)
            # save some memory in eval mode
            if not self.training:
                current_set = current_set.detach()
                current_mask = current_mask.detach()
                repr_loss = repr_loss.detach()
                set_grad = set_grad.detach()
                mask_grad = mask_grad.detach()
            # keep track of intermediates
            intermediate_sets.append(current_set)
            intermediate_masks.append(current_mask)
            repr_losses.append(repr_loss)
            grad_norms.append(set_grad.norm())

        return intermediate_sets, intermediate_masks, repr_losses, grad_norms
