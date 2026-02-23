import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

from config import config
from models.hybrid_nn import QuantaCoreHybridModel
from training.trainer import HybridTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_data():
    """Generates a highly non-linear dataset (Moons) for classification."""
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

def main():
    logger.info("Initializing QuantaCore Hybrid NN Pipeline...")
    train_loader = generate_data()
    
    model = QuantaCoreHybridModel(input_dim=2, output_dim=2)
    trainer = HybridTrainer(model)
    
    logger.info(f"Compute Device: {config.DEVICE} | Quantum Backend: {config.QUANTUM_DEVICE}")
    
    for epoch in range(1, config.EPOCHS + 1):
        loss, acc = trainer.train_epoch(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Training Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
