import os
# pyrefly: ignore [missing-import]
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

os.environ["TK_SILENCE_DEPRECATION"] = "1"

# Optional: torchxrayvision
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except:
    XRV_AVAILABLE = False

# -------------------------------
# Helper Functions for processing
# -------------------------------

def pick_image_file(title="Select an image file"):
    Tk().withdraw()
    path = askopenfilename(title=title)
    if not path:
        raise SystemExit("No image selected.")
    return path

def get_last_conv_layers(model, num_layers=2):
    conv_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    return conv_layers[-num_layers:] if len(conv_layers)>=num_layers else conv_layers

def grad_cam(model, input_tensor, target_layer, class_idx=None):
    model.eval()
    activation = []
    gradient = []

    def forward_hook(_, __, out):
        activation.append(out)

    def backward_hook(_, __, grad_out):
        gradient.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    act = activation[0].detach()
    grad = gradient[0].detach()
    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = torch.sum(weights * act, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    if cam.max()>0:
        cam /= cam.max()

    h1.remove(); h2.remove()
    return cam

def overlay_cam(pil_img, cams, alpha=0.5, size=(224,224)):
    img_resized = pil_img.resize(size)
    img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    overlays = []
    for cam in cams:
        cam_resized = cv2.resize(cam, size)
        heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap, alpha, 0)
        overlays.append(overlay)
    return overlays

def annotate_overlays(overlays, labels):
    annotated = []
    for img, label in zip(overlays, labels):
        img_copy = img.copy()
        cv2.putText(img_copy, label, (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2, cv2.LINE_AA)
        annotated.append(img_copy)
    return annotated

def combine_overlays_grid(overlays_list):
    n = len(overlays_list)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    h, w, _ = overlays_list[0].shape
    grid_img = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
    for idx, overlay in enumerate(overlays_list):
        row = idx // cols
        col = idx % cols
        grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = overlay
    return grid_img

# -------------------------------
# Model Loaders
# -------------------------------

def load_xray_model():
    if not XRV_AVAILABLE:
        raise RuntimeError("torchxrayvision not installed.")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    categories = model.pathologies
    def preprocess(pil_img):
        img = pil_img.convert("L")
        img = np.array(img)
        img = np.expand_dims(img,0)
        img = xrv.datasets.normalize(img,255)
        tx = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = tx(img)
        ten = torch.from_numpy(img).unsqueeze(0).float()
        return ten
    return model, preprocess, categories

def load_skin_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.eval()
    preprocess = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, preprocess, ["benign","malignant"]

def load_brain_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.eval()
    preprocess = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, preprocess, ["no_tumor","tumor"]

# -------------------------------
# Main
# -------------------------------

def main():
    print("Select the model domain:")
    print("1 - Chest X-ray (Top 3 pathologies)")
    print("2 - Skin Cancer")
    print("3 - Brain Tumor")
    choice = input("Enter choice (1/2/3): ").strip()

    img_path = pick_image_file("Select an image")
    pil_img = Image.open(img_path).convert("RGB")
    desktop = os.path.expanduser("~/Desktop")

    if choice=="1":
        model, preprocess, categories = load_xray_model()
        inp = preprocess(pil_img)
        with torch.no_grad():
            logits = model(inp)
            probs = logits.sigmoid()[0].cpu().numpy()
        top3_idx = probs.argsort()[-3:][::-1]
        conv_layers = get_last_conv_layers(model, num_layers=2)
        overlays_all = []
        for class_idx in top3_idx:
            pred_label = categories[class_idx]
            cams=[]
            labels=[]
            for i,(name,layer) in enumerate(conv_layers):
                cam = grad_cam(model, inp, layer, class_idx)
                cams.append(cam)
                labels.append(f"{pred_label} L{i+1}")
            overlays = overlay_cam(pil_img, cams)
            overlays = annotate_overlays(overlays, labels)
            overlays_all.extend(overlays)
        grid_img = combine_overlays_grid(overlays_all)
        out_path = os.path.join(desktop,"gradcam_chest_top3_labeled_dynamic.jpg")
        cv2.imwrite(out_path, grid_img)
        print(f"[OK] Saved Chest X-ray top-3 Grad-CAM grid with layers to {out_path}")
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        plt.axis("off"); plt.title("Chest X-ray Top-3 Grad-CAM (labeled)"); plt.show()

    elif choice=="2":
        model, preprocess, categories = load_skin_model()
        inp = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits,1)[0].cpu().numpy()
        class_idx = int(probs.argmax())
        pred_label = categories[class_idx]
        conv_layers = get_last_conv_layers(model, num_layers=1)
        cams = [grad_cam(model, inp, conv_layers[0][1], class_idx)]
        overlays = overlay_cam(pil_img, cams)
        overlays = annotate_overlays(overlays,[pred_label])
        out_path = os.path.join(desktop,"gradcam_skin_labeled_dynamic.jpg")
        cv2.imwrite(out_path, overlays[0])
        print(f"[OK] Saved Skin Cancer Grad-CAM labeled to {out_path}")
        plt.imshow(cv2.cvtColor(overlays[0], cv2.COLOR_BGR2RGB))
        plt.axis("off"); plt.title(f"{pred_label} Grad-CAM"); plt.show()

    elif choice=="3":
        model, preprocess, categories = load_brain_model()
        inp = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits,1)[0].cpu().numpy()
        class_idx = int(probs.argmax())
        pred_label = categories[class_idx]
        conv_layers = get_last_conv_layers(model, num_layers=1)
        cams = [grad_cam(model, inp, conv_layers[0][1], class_idx)]
        overlays = overlay_cam(pil_img, cams)
        overlays = annotate_overlays(overlays,[pred_label])
        out_path = os.path.join(desktop,"gradcam_brain_labeled_dynamic.jpg")
        cv2.imwrite(out_path, overlays[0])
        print(f"[OK] Saved Brain Tumor Grad-CAM labeled to {out_path}")
        plt.imshow(cv2.cvtColor(overlays[0], cv2.COLOR_BGR2RGB))
        plt.axis("off"); plt.title(f"{pred_label} Grad-CAM"); plt.show()

    else:
        print("Invalid choice.")

if __name__=="__main__":
    main()