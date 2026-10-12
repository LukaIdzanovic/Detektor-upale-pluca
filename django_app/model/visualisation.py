import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from implementation import PneumoniaCNN, CFG, val_test_transforms, THRESHOLD


class Visualisator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path = f"model/best_model_{CFG['model_name']}.pth"
        
        self.model = self.load_model()
        self.model.eval()
        target_layer = self.get_gradcam_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def load_model(self):
        model = PneumoniaCNN(
            model_name=CFG["model_name"], 
            num_classes=CFG["num_classes"], 
            pretrained=False,
            dropout=CFG["dropout"]
        ).to(self.device)
        if os.path.exists(self.best_model_path):
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(f"Upozorenje: Datoteka {self.best_model_path} nije pronađena. Učitane su nasumične težine!")
            
        return model

    def get_gradcam_target_layer(self):
        if "efficientnet" in CFG["model_name"]:
            return self.model.backbone.conv_head
        elif "vgg" in CFG["model_name"]:
            return self.model.backbone.features[-1]
        else:
            return list(self.model.backbone.children())[-2]

    def load_and_preprocess_image(self, image_path):
        pil_image = Image.open(image_path).convert("RGB")
        rgb_img = pil_image.resize((224, 224))
        rgb_img = np.array(rgb_img).astype(np.float32) / 255.0
        transform = val_test_transforms(img_size=224)
        input_tensor = transform(image=np.array(pil_image))["image"]
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        return rgb_img, input_tensor

    @torch.no_grad()
    def predict(self, input_tensor):
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        probability = probs.item()
        prediction = 1 if probability > THRESHOLD else 0
        return prediction, probability

    def generate_cam(self, input_tensor, target_class=0):
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]

    def visualize(self, rgb_img, cam_mask):
        visualization = show_cam_on_image(rgb_img, cam_mask, use_rgb=True)
        return visualization

    def process_image(self, image_path):
        rgb_img, input_tensor = self.load_and_preprocess_image(image_path)
        prediction, probability = self.predict(input_tensor)
        
        cam_mask = self.generate_cam(input_tensor=input_tensor, target_class=0)
        visualization = self.visualize(rgb_img, cam_mask)
        
        label = "PNEUMONIA" if prediction == 1 else "NORMAL"

        return {
            "label": label,
            "probability": probability,
            "original_image": rgb_img,
            "gradcam": visualization
        }