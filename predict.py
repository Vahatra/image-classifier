import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json


def main():

    # Command line argument.
    input = get_input_args()

    image_path = input.image_path
    checkpoint_path = input.checkpoint
    top_k = input.top_k
    category_names = input.category_names
    gpu = input.gpu

    # Category names file.
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load trained model.
    model = load_model(checkpoint_path)

    # Process images, predict classes, and print the flower name and class probability.
    processed_image = process_image(image_path)
    probs, classes = predict(image_path, model, top_k, gpu)
    name = cat_to_name[image_path.split('/')[2]]
    print("Name: {}".format(name))
    print("Probability: {}".format(probs))
    print("Class: {}".format(classes))


def get_input_args():
    """
        Retrieves and parses the command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", type=str, help="Path to image.")
    parser.add_argument("checkpoint", type=str, help="Checkpoint.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Nmber of classes.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="cat_to_name.json file.")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="Use GPU.")

    return parser.parse_args()


def load_model(checkpoint_path):
    """ 
        Loads a model from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)

    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image_path):
    """ 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image_to_process = Image.open(image_path)
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    processed_image = process(image_to_process)

    return processed_image


def predict(image_path, model, topk=5, gpu=True):
    """ 
        Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.cpu()
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    p, c = probs.topk(topk)

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [int(idx_to_class[i]) for i in c[0].numpy()]

    return p[0].detach().numpy(), classes


# Run the program
if __name__ == "__main__":
    main()
