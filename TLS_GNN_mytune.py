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
from LightningMods import TemporalGNN, GCN_LSTM, GNNLightningModule

import json
import time

base_directory = '/gnn'
data_directory = f"{base_directory}/data/"

graph_settings = pd.read_csv(f"{base_directory}/graphs_settings.csv")
graph_settings = graph_settings[-graph_settings.name.str.startswith('TLS_A23_')]
#graph_settings = graph_settings[-graph_settings.name.str.startswith('TLS_A01_')]

'''
hyper_params_dict = {
        'run_name': ['GNN_LSTM', 'A3TGCN2'],
        'model_hidden_dim': [16, 32, 64, 128],
        'GNN_params': {'num_layers': [2, 3], 'dropout': [0.2, 0.3, 0.4]}
        }
'''

model_hidden_dims = [50, 100]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

num_timesteps_in = 48
num_timesteps_out = num_timesteps_in # 24h



def train(net, data_path, run_name):
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


    model = GNNLightningModule(net, train_dataset, test_dataset, batch_size, torch.from_numpy(vde_train_split.edge_index), 
            run_name, sensors, means, stds, device, False)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"{base_directory}/runs/{run_name}/",
        filename='{args.model_type}-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1
        )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=7,min_delta= 0.001)
    stochastic_weight_averager = StochasticWeightAveraging(swa_lrs=1e-2)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(max_epochs=1000,
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, stochastic_weight_averager],
            log_every_n_steps=5,
            accelerator="gpu", devices=1,
            )

    
    # automatically find learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)


    # auto get batch size
    auto_batch_size = tuner.scale_batch_size(model)
    if model.hparams.batch_size > 1000:
        model.hparams.batch_size = 1000

    # update hparams of the model
    model.is_training = True
    trainer.fit(model) 

    # evaluate best model (per training iteration)
    best_model_path = checkpoint_callback.best_model_path
    model = GNNLightningModule.load_from_checkpoint(best_model_path, 
            net=net, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=auto_batch_size, edge_index=torch.from_numpy(vde_train_split.edge_index), 
            run_name=run_name, sensors=sensors, device=device, means=means, stds=stds, is_training=False)
#    model.hparams.batch_size = 100
    final_val_loss = trainer.validate(model)[0]['val_loss']
    end = time.time()

    return final_val_loss, end - start, len(train_dataset), len(test_dataset), auto_batch_size, trainer.current_epoch, os.path.getsize(best_model_path)




graph_settings_results = pd.DataFrame()

for index, graph_setting in graph_settings.iterrows():
    data_path = f"{data_directory}/{graph_setting['name']}"
    for hid_dim in model_hidden_dims:
        net_GNN_LSTM = GCN_LSTM(node_features=2, hidden_dim = hid_dim, num_outputs=2, num_layers=2, dropout=0.3).to(device)
        net_A3TGCN = TemporalGNN(node_features=2, num_outputs = 2, periods=num_timesteps_in, hidden_dim=hid_dim, batch_size=5).to(device)
        net_list = [('GCN_LSTM', net_GNN_LSTM), ('A3TGCN', net_A3TGCN)]
#        net_list = [('A3TGCN', net_A3TGCN)]
        for net_name, net in net_list:
            print( f"{net_name}_{graph_setting['name']}")
            result = train(net, data_path, f"{net_name}_{hid_dim}dims_{graph_setting['name']}")
            net_graph_setting = graph_setting.copy(deep=True)
            net_graph_setting['net_name'], net_graph_setting['hidden_dim'] = net_name, hid_dim
            net_graph_setting['val_loss'], net_graph_setting['running_time'], net_graph_setting['nr_train_samples'], net_graph_setting['nr_test_samples'], net_graph_setting['batch_size'], net_graph_setting['nr_epochs'], net_graph_setting['best_model_size'] = result

            graph_settings_results = pd.concat([graph_settings_results, net_graph_setting.to_frame().T])
            graph_settings_results.to_csv(f"{base_directory}/graph_settings_results.csv", index=False)

            
