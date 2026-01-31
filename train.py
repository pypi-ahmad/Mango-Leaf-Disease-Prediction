import os
import zipfile
import requests
import io
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# --- Configuration ---
DATASET_URL = "https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1"
DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Reduced to 5 for multi-model demo, user said 5-10
BUNDLE_PATH = "multi_model_bundle.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_data():
    """Download and extract dataset."""
    if os.path.exists(DATA_DIR):
        # Check if empty or valid
        if not os.listdir(DATA_DIR):
            print("⚠️ Data directory exists but is empty. Removing...")
            os.rmdir(DATA_DIR)
        else:
            print("✅ Dataset already exists.")
            return

    print("⬇️ Downloading dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        r = requests.get(DATASET_URL)
        if r.status_code == 200:
            print("📦 Extracting...")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(DATA_DIR)
            print("✅ Dataset Ready.")
        else:
            print(f"Error: Status Code {r.status_code}")
            raise Exception("Failed to download dataset")
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        if os.path.exists(DATA_DIR):
             import shutil
             shutil.rmtree(DATA_DIR)
        raise e

def get_data_loaders():
    """Prepare DataLoaders with Transforms."""
    base_path = DATA_DIR
    for root, dirs, files in os.walk(DATA_DIR):
        if "Healthy" in dirs:
            base_path = root
            break
    print(f"📂 Loading images from: {base_path}")

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    full_dataset = datasets.ImageFolder(base_path)
    class_names = full_dataset.classes
    print(f"🌿 Classes: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names

def get_model(model_name, num_classes):
    print(f"🏗️ Building {model_name}...")
    
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Freeze
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace Head
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(DEVICE)

def train_one_model(model_name, model, train_loader, val_loader, class_names):
    criterion = nn.CrossEntropyLoss()
    # Optimize only params that require grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_metrics = {}

    print(f"🚀 Training {model_name} on {DEVICE}...")
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f'  Epoch {epoch+1}/{EPOCHS}', end=' ')
        
        # Train
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        train_acc = train_corrects.double() / len(train_loader.dataset)
        
        # Val
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Metrics
        p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f'- Val Acc: {val_acc:.4f} | F1: {f1:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_metrics = {
                'Accuracy': float(val_acc),
                'Precision': float(p),
                'Recall': float(r),
                'F1-Score': float(f1)
            }

    time_elapsed = time.time() - start_time
    print(f"  🏁 {model_name} finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Best Acc: {best_acc:.4f}")
    
    model.load_state_dict(best_model_wts)
    return model.state_dict(), best_metrics

def main():
    setup_data()
    train_loader, val_loader, class_names = get_data_loaders()
    
    model_names = [
        "EfficientNet-B0",
        "ResNet18", 
        "MobileNetV3-Large",
        "DenseNet121"
    ]
    
    trained_models = {}
    all_metrics = {}
    
    print(f"\n📋 Starting Multi-Model Training for: {model_names}\n")
    
    for name in model_names:
        model = get_model(name, len(class_names))
        state_dict, metrics = train_one_model(name, model, train_loader, val_loader, class_names)
        
        trained_models[name] = state_dict
        all_metrics[name] = metrics
        
        # Clear GPU cache
        del model
        torch.cuda.empty_cache()

    # Save Bundle
    print(f"\n💾 Saving bundle to {BUNDLE_PATH}...")
    bundle = {
        'models': trained_models,
        'class_names': class_names,
        'metrics': all_metrics
    }
    torch.save(bundle, BUNDLE_PATH)
    
    print("\n📊 Final Metrics:")
    for name, m in all_metrics.items():
        print(f"  {name}: {m}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
