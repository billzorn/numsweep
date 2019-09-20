"""Simple convnet for mnist"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from . import modules


ReLULayer = modules.MyReLUModule
# Conv2dLayer = nn.Conv2d
Conv2dLayer = modules.MyConv2dModule
# LinearLayer = nn.Linear
LinearLayer = modules.MyLinearModule


class Net(nn.Module):
    def __init__(self, hidden_features, n_conv1=0, n_conv2=0, n_fc1=0, n_fc2=0):
        super(Net, self).__init__()
        self.conv1 = Conv2dLayer(1, 20, 5, 1, n=n_conv1)
        self.conv2 = Conv2dLayer(20, 50, 5, 1, n=n_conv2)
        self.fc1 = LinearLayer(4*4*50, hidden_features, n=n_fc1)
        self.fc2 = LinearLayer(hidden_features, 10, n=n_fc2)
        self.relu = ReLULayer(n=0)

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
    test = gradcheck(modules.relu, input, eps=1e-6, atol=1e-4)
    print('gradcheck on relu:', test)
    input = (torch.randn(20,20,5,1,dtype=torch.double,requires_grad=True),
             torch.randn(30,20,5,1,dtype=torch.double,requires_grad=True))
    test = gradcheck(modules.conv2d, input, eps=1e-6, atol=1e-4)
    print('gradcheck on conv2d:', test)
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True),
             torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(modules.linear, input, eps=1e-6, atol=1e-4)
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
        params = [20,23,24,25,26]
        #params = [0,10,20,21,22]
        #params = [0,22]
        for conv1 in params:
            for conv2 in params:
                for fc1 in params:
                    for fc2 in params:
                        model = Net(500, conv1, conv2, fc1, fc2).to(device)
                        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

                        print(f'conv1: {conv1}, conv2: {conv2}, fc1: {fc1}, fc2: {fc2}')
                        for epoch in range(1, args.epochs + 1):
                            train(args, model, device, train_loader, optimizer, epoch)
                            loss, correct = test(args, model, device, test_loader, verbose=False)
                            print(f'  epoch {epoch}: {loss:.4f} loss, {correct}/{len(test_loader.dataset)}')



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
    main()
