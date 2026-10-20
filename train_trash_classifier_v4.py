"""
Trash Classification Model - Training & Evaluation Script (v4)
Selection Criterion: Lowest Validation Loss (Min Val Loss).
Configured for 96x96 RGB Input Images across 6 Waste Categories.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import onnx
import onnxruntime as ort
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Target 6 waste categories
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

class TrashClassifierModel(nn.Module):
    """
    Modified Neural Network Architecture for 6-class waste classification.
    Expects input shape: (Batch, 3, 96, 96)
    Features a 2-stage CNN backbone feeding 16 parallel feature processing branches,
    concatenating into a final dense classification head.
    """
    def __init__(self, num_classes=len(CATEGORIES), in_channels=3, num_branches=16, feature_dim=36):
        super(TrashClassifierModel, self).__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim
        
        # 1. Convolutional Backbone
        # Input: (B, 3, 96, 96) -> Conv1 -> (B, 16, 96, 96) -> Pool1 -> (B, 16, 48, 48)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2 -> (B, 16, 48, 48) -> Pool2 -> (B, 16, 24, 24)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling resamples (B, 16, 24, 24) -> (B, 16, 6, 6)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 2. 16 Parallel Processing Branches (Gemm 32x36 removed)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(36, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
            ) for _ in range(num_branches)
        ])
        
        # 3. Dense Transformation & Classification Head
        self.fc1 = nn.Linear(16, 16)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)  # Shape: (Batch, 16, 6, 6)
        
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_branches, self.feature_dim)  # Shape: (Batch, 16, 36)
        
        # Parallel branch forward passes
        branch_outputs = []
        for i in range(self.num_branches):
            b_in = x[:, i, :]
            b_out = self.branches[i](b_in)  # Shape: (Batch, 1)
            branch_outputs.append(b_out)
            
        out = torch.cat(branch_outputs, dim=1)  # Shape: (Batch, 16)
        
        out = self.fc1(out)
        out = self.relu_fc1(out)
        logits = self.fc2(out)  # Shape: (Batch, 6)
        return logits

def find_subfolder(base_dir, candidate_names):
    """Locates train/val subfolders regardless of exact casing or naming."""
    if not base_dir or not os.path.exists(base_dir):
        return None
    for name in candidate_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return None

def get_data_loaders(data_dir=None, train_dir=None, val_dir=None, batch_size=32, img_size=(96, 96)):
    """
    Prepares DataLoaders for 96x96 image inputs.
    """
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    actual_train_dir = train_dir or find_subfolder(data_dir, ['train', 'training', 'Train', 'Training'])
    actual_val_dir = val_dir or find_subfolder(data_dir, ['val', 'validation', 'valid', 'Val', 'Validation', 'Test', 'test'])
    
    if actual_train_dir and actual_val_dir:
        print(f"Loading Training Dataset from:   {actual_train_dir}")
        print(f"Loading Validation Dataset from: {actual_val_dir}")
        
        train_dataset = datasets.ImageFolder(root=actual_train_dir, transform=transform_train)
        val_dataset = datasets.ImageFolder(root=actual_val_dir, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    elif data_dir and os.path.exists(data_dir):
        print(f"Loading single dataset directory from: {data_dir} (automatic 80/20 train/val split)")
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        print("No local dataset directory found. Generating synthetic 96x96 dataset for code verification...")
        num_train_samples = 960
        num_val_samples = 240
        
        X_tr = torch.randn(num_train_samples, 3, *img_size)
        y_tr = torch.randint(0, len(CATEGORIES), (num_train_samples,))
        
        X_va = torch.randn(num_val_samples, 3, *img_size)
        y_va = torch.randint(0, len(CATEGORIES), (num_val_samples,))
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader

def train_model(data_dir=None, train_dir=None, val_dir=None, epochs=15, lr=0.001, batch_size=32, save_dir="./output"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    print(f"Target Categories ({len(CATEGORIES)} classes): {CATEGORIES}")
    print("Input Image Dimensions: 96x96 RGB")
    print("Checkpoint Selection Metric: Lowest Validation Loss (Min Val Loss)\n")
    
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        img_size=(96, 96)
    )
    
    model = TrashClassifierModel(num_classes=len(CATEGORIES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_ckpt_path = os.path.join(save_dir, "best_trash_classifier_v4.pt")
    
    print("Starting Epoch-by-Epoch Training & Evaluation...")
    print("=" * 75)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<16}{'Status':<10}")
    print("=" * 75)
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
        
        # --- EVALUATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        
        status = ""
        # Checkpoint criterion: LOWEST VALIDATION LOSS
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'categories': CATEGORIES,
                'img_size': (96, 96)
            }, best_ckpt_path)
            status = "★ Best"
            
        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<16.2f}{val_loss:<14.4f}{val_acc:<16.2f}{status:<10}")
        
    print("=" * 75)
    print(f"Training Complete! Lowest Validation Loss: {best_val_loss:.4f} achieved at Epoch {best_epoch}\n")
    
    # Load best checkpoint (min val loss) for ONNX export
    best_model = TrashClassifierModel(num_classes=len(CATEGORIES)).to(device)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()
    
    onnx_path = os.path.join(save_dir, "best_trash_classifier_v4.onnx")
    dummy_input = torch.randn(1, 3, 96, 96).to(device)
    
    torch.onnx.export(
        best_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        dynamo=False,
        do_constant_folding=True,
        input_names=['image_input'],
        output_names=['class_logits'],
        dynamic_axes={'image_input': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    
    print(f"Saved Best Checkpoint (Min Val Loss): {best_ckpt_path}")
    print(f"Saved Exported ONNX Model: {onnx_path}")
    
    # Verify exported ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX Model Graph & Runtime Verification PASSED!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Evaluate Trash Classifier v4 (Lowest Val Loss Criterion)")
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder containing 'train' and 'val' subfolders")
    parser.add_argument("--train_dir", type=str, default=None, help="Explicit path to training images subfolder")
    parser.add_argument("--val_dir", type=str, default=None, help="Explicit path to validation images subfolder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./output", help="Directory to save checkpoints & ONNX model")
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
