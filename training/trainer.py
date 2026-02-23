import torch
import torch.nn as nn
import torch.optim as optim
from config import config

class HybridTrainer:
    """Manages the optimization loop and hybrid backpropagation."""
    def __init__(self, model: nn.Module):
        self.model = model.to(config.DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        
        # Adam optimizer updates both classical matrices AND quantum rotation angles
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        
        for data, target in dataloader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / len(dataloader.dataset)
        return avg_loss, accuracy
