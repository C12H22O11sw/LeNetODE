from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models import *


# create advarsarial images with FGSM
def perturb_data(data, epsilon):

    gradient = data.grad.data
    perturbed_data = data + epsilon * gradient.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# train images robustly
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # get data and target from dataloader
        data, target = data.to(device), target.to(device)
        data.requires_grad = True  # ensure that gradients of image are computed (for advarsarial examples)

        # compute loss for standard images
        optimizer.zero_grad()
        output = model(data)
        loss_standard = args.alpha * F.cross_entropy(output, target)
        loss_standard.backward(retain_graph=True)

        # compute loss for advarsarial images
        perturbed_data = perturb_data(data, args.epsilon)
        output = model(perturbed_data)
        loss_adv = (1 - args.alpha) * F.cross_entropy(output, target)
        loss_adv.backward()
        
        # update model
        optimizer.step()
        loss = loss_standard + loss_adv

        # print results
        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}")


def test_standard(model, device, test_loader):
    model.eval()

    # compute accurcy
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print results
    print(f"Standard Test: Accuracy: {correct}/{len(test_loader.dataset)} ({100*correct / len(test_loader.dataset):.0f}%)")

def test_adversarial(args, model, device, test_loader):

    model.eval()
    correct = 0

    # compute accurcy
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # create advarsarial images
        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        perturbed_data = perturb_data(data, args.epsilon)
        output = model(perturbed_data)
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Robust Test set (epsilon={args.epsilon}): Accuracy: {correct}/{len(test_loader.dataset)} ({100*correct / len(test_loader.dataset):.0f}%)")


def main():

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Advarsarial Training of ODENets")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=10, metavar="N",
                        help="input batch size for testing (default: 10)")
    parser.add_argument("--epochs", type=int, default=15, metavar="N",
                        help="number of epochs to train (default: 15)")
    parser.add_argument("--log-interval", type=int, default=1000, metavar="log_interval",
                        help="")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
                        help="learning rate (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=0.1, metavar="epsilon",
                        help="epsilon (default=0.1)")
    parser.add_argument("--uses-ODE", type=bool, default=True, metavar="uses_ODE",
                        help="uses ODE layer (default=True)")
    parser.add_argument("--ode-channels", type=int, default=12, metavar="ode_channels",
                        help="number of channels in ode inception block (default=12)")
    parser.add_argument("--alpha", type=float, default=0.5, metavar="alpha",
                        help="weight of standard images loss relative to advarsarial examples")
    parser.add_argument("--dataset", type=str, default="mnist", metavar="use_mnist",
                        help="")
    args = parser.parse_args()
    
    if args.ode_channels % 4 != 0:
        print("error: --ODE-channels must be a multiple of 4")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    
    if torch.cuda.is_available():
        print("using CUDA")
        device = torch.device("cuda")
        cuda_kwargs = {"num_workers": 1,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        print("using CPU")
        device = torch.device("cpu")



    # load datasets
    mnist_train = datasets.MNIST(root="dataset/", train=True,
                                          transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                          download=True)
    mnist_test = datasets.MNIST(root="dataset/", train=False,
                                         transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                         download=True)

    fashionMNIST_train = datasets.FashionMNIST(root="dataset/", train=True,
                                          transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                          download=True)
    fashionMNIST_test = datasets.FashionMNIST(root="dataset/", train=False,
                                         transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                         download=True)

    # make dataloaders
    if args.dataset == 'mnist':
        train_data = datasets.MNIST(root="dataset/", train=True,
                                          transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                          download=True)
        test_data = datasets.MNIST(root="dataset/", train=False,
                                         transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                         download=True)
    elif args.dataset == 'fashionMNIST':
        train_data = datasets.FashionMNIST(root="dataset/", train=True,
                                          transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                          download=True)
        test_data = datasets.FashionMNIST(root="dataset/", train=False,
                                         transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
                                         download=True)
    else:
        print("error: unknown dataset")
    
    train_loader = torch.utils.data.DataLoader(train_data,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    
    # determine model (ODE+LeNet or just LeNet)
    if args.uses_ODE:
        model = ODE_LeNet(args.ode_channels).to(device)
    else:
        model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train/test model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_standard(model, device, test_loader)
        test_adversarial(args, model, device, test_loader)



if __name__ == "__main__":
    main()