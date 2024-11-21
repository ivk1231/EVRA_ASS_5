import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import CompactMNIST
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Increased batch size for better stability
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactMNIST().to(device)
    
    # Using SGD with momentum and weight decay
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.05,  # Higher initial learning rate
                         momentum=0.9,
                         weight_decay=1e-4)  # Added weight decay
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.05,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.2  # Reach max_lr at 20% of training
    )
    
    # Print model parameters
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count}")
    
    # Training
    model.train()
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        # Update progress bar
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{accuracy:.2f}%',
            'LR': f'{scheduler.get_last_lr()[0]:.5f}'
        })
    
    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    return final_accuracy, param_count

if __name__ == '__main__':
    train() 