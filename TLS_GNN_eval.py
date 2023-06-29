import pandas as pd
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from VDEDatasetLoader import VDEDatasetLoader
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, EarlyStopping
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from LightningMods import TemporalGNN, GCN_LSTM, Baseline, GNNLightningModule

import json
import time


base_directory = '/gnn'
data_directory = f"{base_directory}/data/"

model_hidden_dims = [50, 100]

graph_settings = pd.read_csv(f"{base_directory}/graph_settings_results.csv")
#graph_settings = pd.read_csv(f"{base_directory}/graphs_settings.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_timesteps_in = 48
num_timesteps_out = num_timesteps_in # 24h



def validate(net, data_path, run_name, batch_size, is_baseline):
    sensors, timestamps = np.load(f"{data_path}/sensors.npy", allow_pickle=True), np.load(f"{data_path}/timestamps.npy", allow_pickle=True)

    vde = VDEDatasetLoader(raw_data_dir=f"{data_path}/")
    dataset, means, stds = vde.get_dataset(num_timesteps_in = num_timesteps_in, num_timesteps_out = num_timesteps_out)
    vde_train_split, vde_test_split = temporal_signal_split(dataset, train_ratio=0.8)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ",  len(dataset.features))
    print(next(iter(dataset))) # Show first sample


    train_input = np.array(vde_train_split.features)
    train_target = np.array(vde_train_split.targets)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor) # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor) # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    test_input = np.array(vde_test_split.features)
    test_target = np.array(vde_test_split.targets)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor) # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor) # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    start = time.time()
    
    batch_size = 5 # only inital setting before selecting automatically
    
    if not is_baseline:
        checkpoint_dir = f"{base_directory}/runs/{run_name}"
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".ckpt"):
                checkpoint = (os.path.join(checkpoint_dir, file))
                try:
                    model = GNNLightningModule.load_from_checkpoint(checkpoint,
                            net=net, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=100, edge_index=torch.from_numpy(vde_train_split.edge_index),
                            run_name=run_name, sensors=sensors, device=device, means=means, stds=stds, is_training=False)
                    break
                except:
                    continue

    else: # no pre_loaded baseline
        net = Baseline().to(device)
        model = GNNLightningModule(net, train_dataset, test_dataset, batch_size, torch.from_numpy(vde_train_split.edge_index),
                    run_name, sensors, means, stds, device, False)



    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(max_epochs=1000,
            callbacks=[lr_monitor],
            log_every_n_steps=5,
            accelerator="gpu", devices=1,
            )

    
    trainer.validate(model) 

    final_val_loss = trainer.validate(model)[0]['val_loss']
    end = time.time()

    return final_val_loss


baseline_results = {}

for index, graph_setting in graph_settings.iterrows():
    data_path = f"{data_directory}/{graph_setting['name']}"

    for hid_dim in model_hidden_dims:
        net_GNN_LSTM = GCN_LSTM(node_features=2, hidden_dim = hid_dim, num_outputs=2, num_layers=2, dropout=0.3).to(device)
        net_A3TGCN = TemporalGNN(node_features=2, num_outputs = 2, periods=num_timesteps_in, hidden_dim=hid_dim, batch_size=5).to(device)
        net_list = [('GCN_LSTM', net_GNN_LSTM), ('A3TGCN', net_A3TGCN)]
        for net_name, net in net_list:
            result = validate(net, data_path, f"{net_name}_{graph_setting['name']}", graph_setting['batch_size'], False)

    '''
    result = validate(None, data_path, f"Baseline_{graph_setting['name']}", 100, False)
    baseline_results[graph_setting['name']] = result

    with open("/gnn/baseline_results.json", "w") as outfile:
        json.dump(baseline_results, outfile)
    '''

