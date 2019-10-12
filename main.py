from __future__ import print_function
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# slow, unfortunately; could be vastly sped up with the right C code
import numpy as np
def truncate_significand(t, n):
    if t.dtype == torch.float32:
        a = np.array(t)
        b = np.frombuffer(a.tobytes(), dtype=np.int32)
        b = b & (-1 << n)
        a = np.frombuffer(b.tobytes(), dtype=np.float32).reshape(a.shape)
    else:
        a = np.array(t)
        b = np.frombuffer(a.tobytes(), dtype=np.int64)
        b = b & (-1 << n)
        a = np.frombuffer(b.tobytes(), dtype=np.float64).reshape(a.shape)
    return torch.tensor(a)


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
        return truncate_significand(input.clamp(min=0), n)

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
        return truncate_significand(grad_input, ctx.n), None

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

        return truncate_significand(F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups), n)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        grad_stride = grad_padding = grad_dilation = grad_groups = grad_n = None

        if ctx.needs_input_grad[0]:
            grad_input = truncate_significand(torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups), ctx.n)

        if ctx.needs_input_grad[1]:
            grad_weight = truncate_significand(torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups), ctx.n)

        if bias is not None and ctx.needs_input_grad[2]:
            # I have absolutely no idea if this is even remotely close to being the right thing;
            # However, it produces the right shape, so unlike other things that could go here
            # it doesn't crash.
            grad_bias = truncate_significand(grad_output.sum((0, 2, 3)), ctx.n)

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
        return truncate_significand(output, n)

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
            grad_input = truncate_significand(grad_output.mm(weight), ctx.n)
        if ctx.needs_input_grad[1]:
            grad_weight = truncate_significand(grad_output.t().mm(input), ctx.n)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = truncate_significand(grad_output.sum(0), ctx.n)

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


# select layers used in net

# Conv2dLayer = nn.Conv2d
Conv2dLayer = MyConv2dModule
# LinearLayer = nn.Linear
LinearLayer = MyLinearModule

class Net(nn.Module):
    def __init__(self, hidden_features, n_conv1=0, n_conv2=0, n_fc1=0, n_fc2=0):
        super(Net, self).__init__()
        self.conv1 = Conv2dLayer(1, 20, 5, 1, n=n_conv1)
        self.conv2 = Conv2dLayer(20, 50, 5, 1, n=n_conv2)
        self.fc1 = LinearLayer(4*4*50, hidden_features, n=n_fc1)
        self.fc2 = LinearLayer(hidden_features, 10, n=n_fc2)
        self.relu = MyReLUModule(n=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if args.log_interval and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return test_loss, correct

def check_gradients():
    from torch.autograd import gradcheck
    input = torch.randn(50,dtype=torch.double,requires_grad=True)
    test = gradcheck(relu, input, eps=1e-6, atol=1e-4)
    print('gradcheck on relu:', test)
    # input = (torch.randn(20,20,5,1,dtype=torch.double,requires_grad=True),
    #          torch.randn(30,20,5,1,dtype=torch.double,requires_grad=True))
    # test = gradcheck(conv2d, input, eps=1e-6, atol=1e-4)
    # print('gradcheck on conv2d:', test)
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True),
             torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
    print('gradcheck on linear:', test)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--gradcheck', action='store_true', default=False,
                        help='run gradcheck on custom backwards methods and exit')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sweep', action='store_true', default=False,
                        help='run a parameter sweep')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.gradcheck:
        check_gradients()
        exit(0)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # parameter sweep
    if args.sweep:
        params = [0]
        #params = [0,10,20,21,22]
        #params = [0,22]
        for p in params:
        # for conv2 in params:
        #     for fc1 in params:
        #         for fc2 in params:
            conv1 = 0
            conv2 = 0
            fc1 = 0
            fc2 = 0
            model = Net(500, conv1, conv2, fc1, fc2).to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            print(f'conv1: {conv1}, conv2: {conv2}, fc1: {fc1}, fc2: {fc2}')
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                loss, correct = test(args, model, device, test_loader, verbose=False)
                print(f'  epoch {epoch}: {loss:.4f} loss, {correct}/{len(test_loader.dataset)}')

        return model
                        
        

    else:
        model = Net(500).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    pass
    model = main()
