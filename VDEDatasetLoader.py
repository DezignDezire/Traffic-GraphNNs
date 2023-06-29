import os
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class VDEDatasetLoader(object):

    def __init__(self, raw_data_dir):
    # def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data/METR_LA")):
        super(VDEDatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_data()

    def _read_data(self):
        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose((1, 2, 0)).astype(np.float32)
        # Normalise as in DCRNN paper (via Z-Score Method)
        self.means = np.nanmean(X, axis=(0, 2))
        X = X - self.means.reshape(1, -1, 1)
        self.stds = np.nanstd(X, axis=(0, 2))
        X = X / self.stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, targets = [], []
        ## we exclude samples (times) where data is missing
        valid_indices = []
        samples_with_nans = []
        for pos, (i, j) in enumerate(indices):
            feature = (self.X[:, :, i : i + num_timesteps_in]).numpy()
            target = (self.X[:, :, i + num_timesteps_in : j]).numpy()
            if (np.isnan(feature).any() | np.isnan(target).any()):
                continue
            else:
                features.append(feature)
                targets.append(target)
                valid_indices.append((i,j))
        
        self.features = features
        self.targets = targets
        vaild_indices = np.array(valid_indices)
        run_name = self.raw_data_dir.split('/')[-2]
        np.save(f"/gnn/valid_indices/{run_name}.npy", valid_indices)

    def get_dataset(
            self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
        ) -> StaticGraphTemporalSignal:
            """Returns data iterator for METR-LA dataset as an instance of the
            static graph temporal signal class.

            Return types:
                * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                    forecasting dataset.
            """
            self._get_edges_and_weights()
            self._generate_task(num_timesteps_in, num_timesteps_out)
            dataset = StaticGraphTemporalSignal(
                self.edges, self.edge_weights, self.features, self.targets
            )

            return dataset, self.means, self.stds
