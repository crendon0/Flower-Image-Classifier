#newest iteration
import os
import requests
from pathlib import Path
import tarfile
import json

import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def setup_flower_dataset():
    # defining dataset directory
    data_dir = './flowers'
    FLOWERS_DIR = Path(data_dir)

    # downloading and setting up data if not already present
    if not FLOWERS_DIR.is_dir():
        # creating directory
        FLOWERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Directory created: ./{FLOWERS_DIR}")

        # tarball path
        TARBALL = FLOWERS_DIR / "flower_data.tar.gz"

        # downloading and writing the tarball
        print(f"[INFO] Downloading the file 'flower_data.tar.gz' to ./{FLOWERS_DIR}")
        request = requests.get('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz')
        with open(TARBALL, "wb") as file_ref:
            file_ref.write(request.content)
            print(f"[INFO] 'flower_data.tar.gz' saved to ./{FLOWERS_DIR}")

        # extracting the downloaded tarball
        print(f"[INFO] Extracting the downloaded tarball to ./{FLOWERS_DIR}")
        with tarfile.open(TARBALL, "r:gz") as tar_ref:
            tar_ref.extractall(FLOWERS_DIR)
            print(f"[INFO] 'flower_data.tar.gz' extracted successfully to ./{FLOWERS_DIR}")

        # Fix directory structure - move contents from nested flower_data directory
        flower_data_dir = FLOWERS_DIR / "flower_data"
        if flower_data_dir.exists():
            # Move train, valid, and test directories up one level
            for subdir in ['train', 'valid', 'test']:
                src = flower_data_dir / subdir
                dst = FLOWERS_DIR / subdir
                if src.exists():
                    src.rename(dst)
            
            # Remove now-empty flower_data directory
            flower_data_dir.rmdir()

        # Delete the tarball
        print("[INFO] Deleting the tarball to save space.")
        os.remove(TARBALL)

    # Verify directory structure
    train_dir = FLOWERS_DIR / 'train'
    valid_dir = FLOWERS_DIR / 'valid'
    
    if not train_dir.exists():
        print(f"Warning: Train directory not found at {train_dir}")
    if not valid_dir.exists():
        print(f"Warning: Valid directory not found at {valid_dir}")

    return str(FLOWERS_DIR)


def create_category_json():
    data = {
        "21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster",
        "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy",
        "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly",
        "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist",
        "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower",
        "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation",
        "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone",
        "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow",
        "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid",
        "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia",
        "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura",
        "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium",
        "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily",
        "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william",
        "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon",
        "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula",
        "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower",
        "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple",
        "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus",
        "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily",
        "46": "wallflower", "77": "passion flower", "51": "petunia"
    }

    with open('cat_to_name.json', 'w') as file:
        json.dump(data, file)

def get_args():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    parser.add_argument('data_dir', help='Directory containing the dataset')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='efficientnet_v2_s', choices=['efficientnet_v2_s', 'efficientnet_b0'],
                        help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def main(args):

    print(torch.__version__)
    print(torch.cuda.is_available()) # Should return True when GPU is enabled.

    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    flowers_dir = setup_flower_dataset()

    # Data transforms based on architecture
    if args.arch == 'efficientnet_v2_s':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(384),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(384),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:  # efficientnet_b0
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    # Create proper paths using os.path.join
    train_dir = os.path.join(flowers_dir, 'train')
    valid_dir = os.path.join(flowers_dir, 'valid')

    # Add these debug print statements here
    print("Current working directory:", os.getcwd())
    print("Contents of flowers directory:", os.listdir(args.data_dir))

    print(f"Full train path: {train_dir}")
    print(f"Full valid path: {valid_dir}")

    #verify directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found at {valid_dir}")
        
    print(f"Found training directory at: {train_dir}")
    print(f"Found validation directory at: {valid_dir}")

    # Load datasets using proper paths
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                  for x in ['train', 'valid']}

    # Build model based on architecture choice
    if args.arch == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        classifier_input = model.classifier[1].in_features
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        classifier_input = model.classifier[1].in_features

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # New classifier
    classifier = nn.Sequential(
        nn.Linear(classifier_input, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train model
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        accuracy = 0
        valid_loss = 0

        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    # Save checkpoint
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier
    }

    torch.save(checkpoint, f'{args.save_dir}/checkpoint.pth')

if __name__ == '__main__':
    data_dir = setup_flower_dataset()
    create_category_json()
    args = get_args()
  
    main(args)
