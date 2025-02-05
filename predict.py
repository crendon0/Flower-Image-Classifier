import argparse
import os
import torch
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np

def load_checkpoint(filepath):
    # Check if path exists and try common variations
    if not os.path.exists(filepath):
        # Try prepending checkpoints/ if not found
        alt_path = os.path.join('checkpoints', filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {filepath} or {alt_path}")
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif checkpoint['arch'] == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path, arch):
    """Process an image path into a PyTorch tensor"""
    img = Image.open(image_path)

    # Determine transforms based on model architecture
    if arch == 'efficientnet_v2_s':
        size = 384
        resize = 384
    else:  # efficientnet_b0
        size = 224
        resize = 256

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)

def predict(image_path, model, topk, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.to(device)
    model.eval()

    # Process image and move to device
    img_tensor = process_image(image_path, model.arch)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert output probabilities to softmax probabilities
    ps = torch.exp(output)

    # Get topk probabilities and indices
    top_probs, top_indices = ps.topk(topk, dim=1)

    # Convert indices to actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    return top_probs[0].tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load model and process image
    model = load_checkpoint(args.checkpoint)
    model.arch = args.checkpoint.split('/')[-1].split('_')[0] if '_' in args.checkpoint else 'efficientnet_v2_s'
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    probs, classes = predict(args.image_path, model, args.top_k, device)

    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    # Print results
    print(f"\nTop {args.top_k} predictions for {args.image_path}:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.3f}")

if __name__ == '__main__':
    main()
