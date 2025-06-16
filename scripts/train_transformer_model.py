import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatteryDataset(Dataset):
    def __init__(self, X: np.ndarray, y_soh: np.ndarray, y_soc: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_soh = torch.FloatTensor(y_soh)
        self.y_soc = torch.FloatTensor(y_soc)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_soh[idx], self.y_soc[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        
        self.soh_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.soc_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        
        soh_pred = torch.sigmoid(self.soh_head(x))
        soc_pred = torch.sigmoid(self.soc_head(x))
        
        return soh_pred, soc_pred

class TransformerTrainer:
    def __init__(self, model_dir: str = "models/transformer"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_model(self, 
                  X_train: np.ndarray, 
                  y_soh_train: np.ndarray, 
                  y_soc_train: np.ndarray,
                  X_val: np.ndarray = None,
                  y_soh_val: np.ndarray = None,
                  y_soc_val: np.ndarray = None,
                  epochs: int = 50,
                  batch_size: int = 32,
                  learning_rate: float = 1e-4) -> None:
        """Train the Transformer model."""
        logger.info("Starting Transformer model training")
        
        # Create datasets and dataloaders
        train_dataset = BatteryDataset(X_train, y_soh_train, y_soc_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = BatteryDataset(X_val, y_soh_val, y_soc_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = TransformerModel(input_dim=X_train.shape[-1]).to(self.device)
        
        # Loss functions and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x, y_soh, y_soc = [t.to(self.device) for t in batch]
                
                optimizer.zero_grad()
                soh_pred, soc_pred = self.model(x)
                
                loss = 0.5 * criterion(soh_pred, y_soh.unsqueeze(1)) + \
                      0.5 * criterion(soc_pred, y_soc.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        x, y_soh, y_soc = [t.to(self.device) for t in batch]
                        soh_pred, soc_pred = self.model(x)
                        
                        loss = 0.5 * criterion(soh_pred, y_soh.unsqueeze(1)) + \
                              0.5 * criterion(soc_pred, y_soc.unsqueeze(1))
                        
                        val_loss += loss.item()
                    
                val_loss /= len(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_dir / 'transformer_model.pth')
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping triggered")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
                
        logger.info("Model training completed successfully")
        
    def evaluate_model(self, 
                     X_test: np.ndarray, 
                     y_soh_test: np.ndarray, 
                     y_soc_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        test_dataset = BatteryDataset(X_test, y_soh_test, y_soc_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            soh_mse = 0.0
            soc_mse = 0.0
            
            for batch in test_loader:
                x, y_soh, y_soc = [t.to(self.device) for t in batch]
                soh_pred, soc_pred = self.model(x)
                
                soh_mse += criterion(soh_pred, y_soh.unsqueeze(1)).item()
                soc_mse += criterion(soc_pred, y_soc.unsqueeze(1)).item()
            
            soh_mse /= len(test_loader)
            soc_mse /= len(test_loader)
            
            metrics = {
                'soh_mse': soh_mse,
                'soc_mse': soc_mse
            }
            
            logger.info(f"Test metrics: {metrics}")
            return metrics

def main():
    """Main function to train Transformer model."""
    # Load preprocessed data
    preprocessor = BatteryDataPreprocessor()
    (X_train, X_val, X_test), (y_soh_train, y_soh_val, y_soh_test), \
    (y_soc_train, y_soc_val, y_soc_test) = preprocessor.prepare_data_for_training()
    
    # Initialize trainer
    trainer = TransformerTrainer()
    
    # Train model
    trainer.train_model(
        X_train=X_train,
        y_soh_train=y_soh_train,
        y_soc_train=y_soc_train,
        X_val=X_val,
        y_soh_val=y_soh_val,
        y_soc_val=y_soc_val
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(
        X_test=X_test,
        y_soh_test=y_soh_test,
        y_soc_test=y_soc_test
    )

if __name__ == "__main__":
    main()
