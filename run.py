import pytorch_lightning as pl
import os
import torch

from mutag.src.models import GraphLevelGNN 
import torch_geometric
import torch_geometric.data as geom_data
from pytorch_lightning.callbacks import ModelCheckpoint

tu_dataset = torch_geometric.datasets.TUDataset(root='./data', name="MUTAG")

torch.manual_seed(42)
tu_dataset.shuffle()
train_dataset = tu_dataset[:150]
test_dataset = tu_dataset[150:]

graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=64) # Additional loader if you want to change to a larger dataset
graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def train_graph_classifier(model_name, **model_kwargs):
    pl.seed_everything(42)
    root_dir = os.path.join('./gnnmodel', 'GraphLevel' + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir = root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc')],
        gpus = 1 if str(device).startswith('cuda') else 0,
        max_epochs = 500,
        progress_bar_refresh_rate = 0
    )
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join('./gnnmodel', 'GraphLevel%s.ckpt' % model_name)
    if os.path.isfile(pretrained_filename):
        print('Found pretrained model, loading...')
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(
            c_in=tu_dataset.num_node_features,
            c_out=1 if tu_dataset.num_classes==2 else tu_dataset.num_classes,
            **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
    train_result = trainer.test(model, test_dataloaders=graph_train_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=graph_test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    return model, result

model, result = train_graph_classifier(model_name="GraphConv",
                                       c_hidden=256,
                                       layer_name="GraphConv",
                                       num_layers=3,
                                       dp_rate_linear=0.5,
                                       dp_rate=0.0)

print("Train performance: %4.2f%%" % (100.0*result['train']))
print("Test performance:  %4.2f%%" % (100.0*result['test']))