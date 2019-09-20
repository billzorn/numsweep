"""Converted pytorch operators and modules."""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, n=0, n_backward=None):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.n = n if n_backward is None else n_backward
        return utils.truncate_significand(input.clamp(min=0), n)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return utils.truncate_significand(grad_input, ctx.n), None

# relu = F.relu
relu = MyReLU.apply

class MyReLUModule(nn.modules.Module):
    __constants__ = ['n']

    def __init__(self, n):
        self.n = n
        super().__init__()

    def forward(self, input):
        return relu(input, self.n)


# stolen from the webz, doesn't crash, it's possible it's correct.
class MyConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, n=0, n_backward=None):

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.n = n if n_backward is None else n_backward

        return utils.truncate_significand(F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups), n)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        grad_stride = grad_padding = grad_dilation = grad_groups = grad_n = None

        if ctx.needs_input_grad[0]:
            grad_input = utils.truncate_significand(nn.grad.conv2d_input(input.shape, weight, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups), ctx.n)

        if ctx.needs_input_grad[1]:
            grad_weight = utils.truncate_significand(nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups), ctx.n)

        if bias is not None and ctx.needs_input_grad[2]:
            # I have absolutely no idea if this is even remotely close to being the right thing;
            # However, it produces the right shape, so unlike other things that could go here
            # it doesn't crash.
            grad_bias = utils.truncate_significand(grad_output.sum((0, 2, 3)), ctx.n)

        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, grad_n

# conv2d = F.conv2d
conv2d = MyConv2d.apply

# from torch.nn.modules.conv
class MyConv2dModule(nn.modules.conv._ConvNd):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'n']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', n=0):
        self.n = n
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        padding = nn.modules.utils._pair(padding)
        dilation = nn.modules.utils._pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, nn.modules.utils._pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return conv2d(F.pad(input, expanded_padding, mode='circular'),
                          weight, self.bias, self.stride,
                          _pair(0), self.dilation, self.groups, self.n)
        return conv2d(input, weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups, self.n)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)


# from https://pytorch.org/docs/stable/notes/extending.html
class MyLinear(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, n=0, n_backward=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.n = n if n_backward is None else n_backward
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return utils.truncate_significand(output, n)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_n = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = utils.truncate_significand(grad_output.mm(weight), ctx.n)
        if ctx.needs_input_grad[1]:
            grad_weight = utils.truncate_significand(grad_output.t().mm(input), ctx.n)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = utils.truncate_significand(grad_output.sum(0), ctx.n)

        return grad_input, grad_weight, grad_bias, grad_n

# linear = F.linear
linear = MyLinear.apply

# from torch.nn.modules.linear
class MyLinearModule(nn.modules.Module):
    __constants__ = ['bias', 'in_features', 'out_features', 'n']

    def __init__(self, in_features, out_features, bias=True, n=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.n = n
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return linear(input, self.weight, self.bias, self.n)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
