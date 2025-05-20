import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

IMG_SIZE = 128
MODEL_PATH = 'cnn_brain_tumor.pth'

class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumorCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
        class_names = ['No Tumor', 'Tumor']
        return class_names[pred], float(probs[pred])

def grad_cam(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumorCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    # Hook for gradients
    gradients = []
    activations = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    def forward_hook(module, input, output):
        activations.append(output.detach())
    # Register hooks on last conv layer
    handle_f = model.features[-3].register_forward_hook(forward_hook)
    handle_b = model.features[-3].register_backward_hook(backward_hook)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()
    # Grad-CAM calculation
    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    handle_f.remove()
    handle_b.remove()
    return img_np, overlay 