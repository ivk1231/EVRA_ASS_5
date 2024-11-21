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
    
    # Load MNIST dataset with enhanced augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=1)
    
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
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        # Update progress bar
        accuracy = 100. * correct / total
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 
                                  'Accuracy': f'{accuracy:.2f}%'})
    
    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    return final_accuracy, param_count

if __name__ == '__main__':
    train() 