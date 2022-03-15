import pytorch_lightning as pl
from typing import Optional
from torch_geometric import datasets
from torch_geometric.data import DataLoader as GraphDataLoader

class MUTAGDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.TUDataset(root=self.data_dir, name="MUTAG")
    
    def setup(self, stage: Optional[str]):
        self.mutag = datasets.TUDataset(root=self.data_dir, name="MUTAG")
        self.mutag.shuffle()
        
        self.train_dataset = self.mutag[:150]
        self.val_dataset = self.mutag[150:]
    
    def train_dataloader(self):
        return GraphDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return GraphDataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return GraphDataLoader(self.val_dataset, batch_size=self.batch_size)
