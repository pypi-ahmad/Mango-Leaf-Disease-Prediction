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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Configuration ---
DATASET_URL = "https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1"
DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Reduced to 5 for multi-model demo, user said 5-10
BUNDLE_PATH = "multi_model_bundle.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def safe_extract_zip(zip_file, target_dir):
    target_dir_abs = os.path.abspath(target_dir)
    for member in zip_file.namelist():
        member_path = os.path.abspath(os.path.join(target_dir_abs, member))
        if not member_path.startswith(target_dir_abs + os.sep) and member_path != target_dir_abs:
            raise ValueError(f"Unsafe path in archive: {member}")
    zip_file.extractall(target_dir)

def setup_data():
    """Download and extract dataset."""
    if os.path.exists(DATA_DIR):
        subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if subdirs:
            print("✅ Dataset already exists.")
            return
        else:
            print("⚠️ Data directory exists but contains no subdirectories. Removing...")
            import shutil
            shutil.rmtree(DATA_DIR)

    print("⬇️ Downloading dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        r = requests.get(DATASET_URL, timeout=60)
        if r.status_code == 200:
            print("📦 Extracting...")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                safe_extract_zip(z, DATA_DIR)
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

    if not os.path.isdir(base_path):
        raise ValueError(f"Dataset root not found: {base_path}")

    class_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not class_dirs:
        raise ValueError(f"No class folders found in dataset root: {base_path}")

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

    if len(full_dataset) < 3:
        raise ValueError("Dataset too small for train/val/test split.")

    generator = torch.Generator().manual_seed(SEED)
    shuffled_indices = torch.randperm(len(full_dataset), generator=generator).tolist()

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    if val_size == 0:
        val_size = 1
    test_size = len(full_dataset) - train_size - val_size
    if test_size <= 0:
        test_size = 1
        train_size = len(full_dataset) - val_size - test_size
        if train_size <= 0:
            raise ValueError("Dataset too small after split adjustment.")

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    train_dataset = datasets.ImageFolder(base_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(base_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(base_path, transform=val_transform)

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names

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

def _evaluate_loader_metrics(model, data_loader):
    criterion = nn.CrossEntropyLoss()
    total_correct = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            total_correct += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = total_correct.double() / len(data_loader.dataset)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'Accuracy': float(acc),
        'Precision': float(p),
        'Recall': float(r),
        'F1-Score': float(f1)
    }


def train_one_model(model_name, model, train_loader, val_loader):
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
        train_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_corrects += torch.sum(preds == labels.data)
            
        train_acc = train_corrects.double() / len(train_loader.dataset)
        
        # Val
        model.eval()
        val_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
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
    try:
        train_loader, val_loader, test_loader, class_names = get_data_loaders()
    except Exception as e:
        print(f"❌ Failed to prepare data loaders: {e}")
        return
    
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
        state_dict, _ = train_one_model(name, model, train_loader, val_loader)

        model.load_state_dict(state_dict)
        metrics = _evaluate_loader_metrics(model, test_loader)
        
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
