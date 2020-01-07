import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image


# Pretrained models
vgg19 = models.vgg19(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

class_to_idx = []
data_transforms = {}
image_datasets = {}
dataloaders = {}


def main():

    # Command line argument.
    input = get_input_args()

    data_dir = input.data_dir
    save_dir = input.save_dir
    arch = input.arch
    learning_rate = input.learning_rate
    hidden_units = input.hidden_units
    epochs = input.epochs
    gpu = input.gpu

    # Load and process images from data directory.
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    load(train_dir, test_dir, valid_dir)

    # Build the network.
    model = vgg16
    if arch == "vgg19":
        model = vgg19

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier

    epochs = epochs
    learning_rate = learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train
    train(model, epochs, learning_rate, criterion, optimizer, gpu)

    # Validate
    validate(model, criterion, gpu)

    # Save Checkpoint
    model.class_to_idx = image_datasets['training'].class_to_idx
    torch.save({'arch': arch,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
               save_dir)

# Function Definitions


def get_input_args():
    """
        Retrieves and parses the command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="Data directory.")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth",
                        help="Save directory.")
    parser.add_argument("--arch", type=str, default="vgg16",
                        help="Architecture: vgg16, vgg19.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--hidden_units", type=int, default=4096,
                        help="Hidden layers.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epoch number.")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="Use GPU.")

    return parser.parse_args()


def load(train_dir, test_dir, valid_dir):
    """
        Loads and transforms data.
    """
    global data_transforms, image_datasets, dataloaders
    data_transforms = {
        'training': transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),

        'validation': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

        'testing': transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }

    class_to_idx = image_datasets['training'].class_to_idx

    return dataloaders['training'], dataloaders['testing'], dataloaders['validation']


def train(model, epochs, learning_rate, criterion, optimizer, gpu=True):
    """
        Train the network.
    """
    training_loader = dataloaders['training']
    validation_loader = dataloaders['validation']
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    model.train()
    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                loss = 0

                for inputs, labels in iter(validation_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model.forward(inputs)
                    loss += criterion(output, labels).item()
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                loss, accuracy = loss / \
                    len(validation_loader), accuracy/len(validation_loader)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(
                          running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))


def validate(model, criterion, gpu=True):
    """
        Validation on the test set.
    """
    testing_loader = dataloaders['testing']
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    for inputs, labels in iter(testing_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))


# Run the program
if __name__ == "__main__":
    main()
