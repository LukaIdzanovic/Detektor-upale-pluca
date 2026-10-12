import os
import random
import numpy as np
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CFG = {
    "model_name": "tf_efficientnet_b0_ns",  #"vgg16"
    "num_classes": 1,
    "pretrained": True,
    "dropout": 0.3,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 20
}

THRESHOLD = 0.45

train_data_path = "..\\rsna\\train"
validation_data_path = "..\\rsna\\val"
test_data_path_kaggle = "..\\kaggle\\test"
test_data_path_rsna = "..\\rsna\\test"

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        classes = ["NORMAL", "PNEUMONIA"]
        for label, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            if not os.path.exists(class_path):
                continue
            for img_name in os.listdir(class_path):
                if img_name.startswith('.'):
                    continue
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

def train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_test_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def create_train_dataloader(batch_size=32):
    return DataLoader(ChestXrayDataset(train_data_path, transform=train_transforms()), batch_size=batch_size, shuffle=True, num_workers=1)

def create_val_dataloader(batch_size=32):
    return DataLoader(ChestXrayDataset(validation_data_path, transform=val_test_transforms()), batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

def create_test_dataloader(test_data_path, batch_size=32):
    return DataLoader(ChestXrayDataset(test_data_path, transform=val_test_transforms()), batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

class PneumoniaCNN(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, dropout=0.3):
        super(PneumoniaCNN, self).__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        if "efficientnet" in self.model_name and hasattr(self.backbone, "blocks"):
            for p in self.backbone.blocks[-1].parameters():
                p.requires_grad = True
        elif "vgg" in self.model_name and hasattr(self.backbone, "features"):
            for p in self.backbone.features[-5:].parameters():
                p.requires_grad = True
        else:
            for p in list(self.backbone.children())[-1].parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

def train_epoch(model, scaler, device, criterion, optimizer, data_loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm.tqdm(data_loader, desc="Training")
    
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = (torch.sigmoid(logits) > THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, criterion, device, data_loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for imgs, labels in tqdm.tqdm(data_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        preds = (torch.sigmoid(logits) > THRESHOLD).float()
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    return {
        "loss": total_loss / len(data_loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = create_train_dataloader()
    val_loader = create_val_dataloader()
    test_loader_kaggle = create_test_dataloader(test_data_path=test_data_path_kaggle)
    test_loader_rsna = create_test_dataloader(test_data_path=test_data_path_rsna)

    train_dataset = ChestXrayDataset(train_data_path)
    counts = Counter(train_dataset.labels)
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = PneumoniaCNN(CFG["model_name"], CFG["num_classes"], CFG["pretrained"], CFG["dropout"]).to(device)
    model.freeze_backbone()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    patience, patience_counter = 5, 0
    model_save_filename = f"best_model_{CFG['model_name']}.pth"

    for epoch in range(CFG["epochs"]):
        print(f"\n======== EPOCH {epoch+1}/{CFG['epochs']} ========")
        
        if epoch == 3:
            print("Unfreezing last block...")
            model.unfreeze_last_block()
            optimizer.param_groups.clear()
            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1e-5, 'weight_decay': CFG["weight_decay"]})

        if epoch == 8:
            print("Unfreezing all...")
            model.unfreeze_all()
            optimizer.param_groups.clear()
            optimizer.add_param_group({'params': model.parameters(), 'lr': 5e-6, 'weight_decay': CFG["weight_decay"]})

        train_loss, train_acc = train_epoch(model, scaler, device, criterion, optimizer, train_loader)
        val_metrics = evaluate(model, criterion, device, val_loader)
        scheduler.step(val_metrics["loss"])

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"Val Loss   : {val_metrics['loss']:.4f} | Val Acc   : {val_metrics['accuracy']:.4f}")
        print(f"Precision  : {val_metrics['precision']:.4f} | Recall    : {val_metrics['recall']:.4f}")
        print(f"F1 Score   : {val_metrics['f1']:.4f}")
        print("Confusion Matrix:\n", val_metrics["confusion_matrix"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), model_save_filename)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nLoading best model: {model_save_filename}")
    model.load_state_dict(torch.load(model_save_filename))

    print("\n==============================\nKAGGLE TEST SET\n==============================")
    kaggle_metrics = evaluate(model, criterion, device, test_loader_kaggle)
    print(f"Loss      : {kaggle_metrics['loss']:.4f}\nAccuracy  : {kaggle_metrics['accuracy']:.4f}")
    print(f"Precision : {kaggle_metrics['precision']:.4f}\nRecall    : {kaggle_metrics['recall']:.4f}")
    print(f"F1 Score  : {kaggle_metrics['f1']:.4f}\nConfusion Matrix:\n", kaggle_metrics["confusion_matrix"])

    print("\n==============================\nRSNA TEST SET\n==============================")
    rsna_metrics = evaluate(model, criterion, device, test_loader_rsna)
    print(f"Loss      : {rsna_metrics['loss']:.4f}\nAccuracy  : {rsna_metrics['accuracy']:.4f}")
    print(f"Precision : {rsna_metrics['precision']:.4f}\nRecall    : {rsna_metrics['recall']:.4f}")
    print(f"F1 Score  : {rsna_metrics['f1']:.4f}\nConfusion Matrix:\n", rsna_metrics["confusion_matrix"])

if __name__ == "__main__":
    main()