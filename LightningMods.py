import random
import os
import pandas as pd
import numpy as np
import plotly.express as pe

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(42)
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2, TGCN2



class Baseline(torch.nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, x, edge_index):
        return x


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, num_outputs, periods, batch_size):
        super(TemporalGNN, self).__init__()
        self.hidden_dim = hidden_dim
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(
                          in_channels=node_features,
                          out_channels=hidden_dim,
                          periods=periods,
                          batch_size=batch_size,
                          improved=True)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods * num_outputs)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """

        h = self.tgnn(x, edge_index)
        h = F.relu(h)

        out = self.linear(h)
        out = out.view(x.shape)
        return out



class GCN_LSTM(pl.LightningModule):
    def __init__(self, node_features, hidden_dim, num_outputs, num_layers, dropout):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, num_outputs)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features, num_timesteps = x.shape
        # Reshape for LSTM
        x = x.view(batch_size * num_nodes, num_features, num_timesteps)
        # Reshape for GCNConv: (batch_size * num_nodes, num_timesteps, num_features)
        x = x.permute(0, 2, 1)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Reshape LSTM output
        lstm_out = lstm_out.reshape(-1, self.hidden_dim)

        out = self.fc(lstm_out)
        out = out.view(batch_size, num_nodes, num_timesteps, self.num_outputs)
        out = out.permute(0, 1, 3, 2)
        return out





class GNNLightningModule(pl.LightningModule):

    def __init__(self, net: torch.nn.Module, train_dataset, test_dataset, batch_size, edge_index, run_name, sensors, means, stds, device, is_training=True):
        super(GNNLightningModule, self).__init__()
        self.save_hyperparameters(ignore=['net', 'train_dataset', 'test_dataset'])
        self.net = net
        self.run_name = run_name
        self.sensors = sensors
        self.means = means
        self.stds = stds
        self.train_dataset = train_dataset
        self.test_dataset =  test_dataset
        self.batch_size = batch_size
        self.edge_index = edge_index.to(device)
        self.is_training = is_training
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 0
        self.scheduler = None

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        encoder_inputs, labels = batch
        y_hat = self.net(encoder_inputs, self.edge_index)
        loss = self.loss_fn(y_hat, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        encoder_inputs, labels = batch
        y_hat = self.net(encoder_inputs, self.edge_index)
        loss = self.loss_fn(y_hat, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        if not self.is_training:
        #if batch_idx == 0:
            self.plot_eval(batch_idx, encoder_inputs, labels, y_hat)
        
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics['val_loss']

        if self.is_training:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr)
            self.scheduler.step(val_loss)
        print(f"Validation loss: {val_loss:.4f}")
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=4, verbose=True)
        return optimizer

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')

    def train_dataloader(self):
        print('reload train loader:', self.hparams.batch_size)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=32, pin_memory=True)

    def plot_eval(self, batch_idx, encoder_inputs, labels, y_hat, plots_path='eval_plots'):

        eval_df = pd.DataFrame()
        for r_i in range(len(encoder_inputs)):
            x, y, y_pred = encoder_inputs[r_i], labels[r_i], y_hat[r_i]
            x, y, y_pred = self.retransform_data(x.cpu().numpy()), self.retransform_data(y.cpu().numpy()), self.retransform_data(y_pred.cpu().numpy())

            n_sensors = x[:,0,:].shape[0]
            n_timesteps = x[:,0,:].shape[1]

            for i in range(2):
                x_df = pd.DataFrame(x[:,i,:].T, index=np.array(range(n_timesteps))-n_timesteps, columns=self.sensors)
                x_df["type"] = "x"
                y_df = pd.DataFrame(y[:,i,:].T, index=np.array(range(n_timesteps)), columns=self.sensors)
                y_df["type"] = "y"
                y_pred_df = pd.DataFrame(y_pred[:,i,:].T, index=np.array(range(n_timesteps)), columns=self.sensors)
                y_pred_df["type"] = "y_pred"

                plot_df = pd.concat([x_df, y_df, y_pred_df])
            

                plot_df.index = plot_df.index.set_names(['timestamp'])
                plot_df = plot_df.reset_index()
                melt_df = pd.melt(plot_df, id_vars=['timestamp', 'type'], value_vars=self.sensors, var_name='sensor')
                melt_df['batch'] = r_i
                melt_df['score'] = 'flow' if i == 0 else 'velocity'
                eval_df = pd.concat([eval_df, melt_df])

        eval_df.to_csv(f"/gnn/outputs/{self.run_name}_{self.net.hidden_dim}_batch{batch_idx}.csv")


            # fig = pe.line(melt_df, x='timestamp', y='value', facet_col="sensor", facet_col_wrap=4, color="type", width = 1500, height = 1500)
            # fig.show()

            #dir_path = f"/gnn/eval_plots/epoch-{self.current_epoch}/"
            #if not os.path.exists(dir_path):
            #    os.mkdir(dir_path)
            #fig.write_image(f"{dir_path}batch-{r_i}.png")




    def retransform_data(self, X_scaled):
        X_retransformed = X_scaled * self.stds.reshape(1, -1, 1) + self.means.reshape(1, -1, 1)
        return X_retransformed
