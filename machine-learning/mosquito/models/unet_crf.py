from pyexpat import features
import pandas as pd
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import tqdm
from matplotlib import pyplot as plt
from positional_encodings.torch_encodings import PositionalEncoding1D
from focal_loss import FocalLoss
import wandb
from torch_struct import LinearChainCRF

class Model():
    def __init__(self, model: "UNet1D",
                 ignore_N=None, 
                 save_path=None,
                 data_columns = None,
                 label_map= None,
                 seed=42,
                 enable_wandb_logging=True):
        random.seed(seed) 
        self.enable_wandb_logging = enable_wandb_logging
        if data_columns == None:
            data_columns = ["post_rect"],
        if label_map == None:
            label_map= {"J"  : 0, "K"  : 1, "L"  : 2, "M"  : 3, "N"  : 4, "W"  : 5},

        self.label_map = label_map
        self.inv_label_map = {i:label for label, i in self.label_map.items()}

        self.data_columns = data_columns
        self.ignore_N = ignore_N


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.label_map)
        self.model = model

        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def train(self, probes, test_probes, **kwargs):
        self.model = self.model.to(self.device)
        
        batch_size = kwargs.get("batch_size", 1)
        epochs = kwargs.get("epoches", 128)
        lr = kwargs.get("lr", 5e-4)
        weight_decay = kwargs.get("weight_decay", 1e-7)
        folds = kwargs.get("folds", 5)
        save_train_curve = kwargs.get("save_train_curve", False)
        show_train_curve = kwargs.get("show_train_curve", False)
        augment = kwargs.get("augment", True)
        augment_factor = kwargs.get("augment_factor", 1)
        loss_gamma = kwargs.get('loss_gamma', 1.5)
        loss_alpha = kwargs.get('loss_alpha', None)
        

        if self.enable_wandb_logging and wandb.run is not None:
            wandb.watch(self.model, log="all", log_freq=10, log_graph=True)

        tr_dfs, tr_df = self.load_probes(probes)
        tr_dataset = TimeSeriesDataset(tr_dfs, self.label_map, data_columns=self.data_columns, 
                                       class_column = "labels", ignore_N=self.ignore_N)
        tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

        if test_probes:
            test_dfs, test_df = self.load_probes(test_probes)
            test_dataset = TimeSeriesDataset(test_dfs, self.label_map, data_columns=self.data_columns,
                                             class_column = "labels",ignore_N=self.ignore_N)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, capturable=False)

        train_losses = []
        test_losses = []
        pbar = tqdm.tqdm(range(epochs), desc=f"Fold {folds} Training")
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for batch in tr_dataloader:
                x, y, weights = batch
                x, y, weights = x.to(self.device), y.to(self.device), \
                                weights.to(self.device)

                optimizer.zero_grad()
                crf = self.model(x.permute(0,2,1))
                loss = crf.nll(y)

                weighted_loss = (loss) * weights
                weighted_loss = weighted_loss.mean()
                weighted_loss.backward()
                optimizer.step()
                
                running_loss += weighted_loss.item()
            train_loss = running_loss / len(tr_dataloader)
            train_losses.append(train_loss)

            # Get the test loss
            if test_probes:
                self.model.eval()
                with torch.no_grad():
                    running_loss = 0
                    for batch in test_dataloader:
                        x, y, weights = batch
                        x, y, weights = x.to(self.device), y.to(self.device), \
                                        weights.to(self.device)
                        
                        crf = self.model(x.permute(0,2,1))
                        loss = crf.nll(y)
                        
                        weighted_loss = loss * weights
                        weighted_loss = weighted_loss.mean()
                        running_loss += weighted_loss.item()
                    test_loss = running_loss / len(test_dataloader)
                    test_losses.append(test_loss)
                pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "test_loss": f"{test_loss:.4f}" if test_probes else "N/A"})

            if self.enable_wandb_logging and wandb.run is not None:
                wandb.log({"train_loss": train_loss, "val_loss": test_loss if test_probes else None}, step=epoch)

        if save_train_curve:
            plt.plot(train_losses, label = "Train")
            plt.plot(test_losses, label = "Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(f"{self.save_path}/loss_curve_fold{self.fold}.png")
        if show_train_curve:
            plt.plot(train_losses, label = "Train")
            plt.plot(test_losses, label = "Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()

    def predict(self, probes, batch_size, preprocess = False, return_logits=False):
        test_dataset = TimeSeriesDataset(probes, self.label_map, data_columns=self.data_columns,
                                         class_column = "labels", ignore_N=self.ignore_N)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
        all_predictions = []
        all_logits = []
        self.model.eval()
        with torch.no_grad():
            for probe in test_dataloader:
                x, _, _ = probe
                x = x.to(self.device)

                #outputs = self.model(x.permute(0,2,1)).permute(0,2,1).reshape(1, -1, self.num_classes, self.num_classes)[:, 1:].contiguous()
                crf = self.model(x.permute(0,2,1))
                if return_logits:
                    all_logits.append(crf.logits().cpu())

                outputs = crf.argmax().view(-1).cpu().tolist()
                output_labels = [self.inv_label_map[x] for x in outputs]
                all_predictions.append(output_labels)
            if return_logits:
                return all_predictions, all_logits
            else:
                return all_predictions
            
    def predict_proba(self, probes, preprocess = False):
        test_dataset = TimeSeriesDataset(probes, self.label_map, data_columns=self.data_columns,
                                         class_column = "labels", ignore_N=self.ignore_N)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        all_probabilities = []
        self.model.eval()
        with torch.no_grad():
            for probe in test_dataloader:
                x, _, _ = probe
                x = x.to(self.device)

                crf = self.model.forward(x.permute(0,2,1))
                all_probabilities.append(crf.marginals().cpu())
        return all_probabilities

    def load_probes(self, probes):
        big_probe = pd.concat(probes, axis=0)
        return probes, big_probe

    def save(self):
        torch.save(self.model.state_dict(), "unet_weights")

    # def load(self, path = None):
    #     self.model = UNet1D(input_size=len(self.data_columns), 
    #                         output_size=self.num_classes,
    #                         growth_factor=self.growth_factor,
    #                         features=self.features,
    #                         num_layers=self.num_layers, 
    #                         n_conv_steps_per_block=self.n_conv_steps_per_block, 
    #                         dropout_rate=self.dropout_rate, 
    #                         block_kernel_size=self.block_kernel_size,
    #                         up_down_sample_kernel_size=self.up_down_sample_kernel_size,
    #                         block_padding=self.block_padding,
    #                         bottleneck_type=self.bottleneck_type, 
    #                         transformer_window_size=self.transformer_window_size, 
    #                         embed_dim=self.embed_dim, 
    #                         transformer_layers=self.transformer_layers, 
    #                         transformer_nhead=self.transformer_nhead,

    #                         ) 
    #     self.model.load_state_dict(torch.load(path, weights_only=True, map_location = self.device))
    #     self.model = self.model.to(self.device)

class TimeSeriesDataset(Dataset):
    def __init__(self, dfs, label_map, data_columns, class_column, transform=None, weight=False, ignore_N=False):
        """
        dfs: List of DataFrames, each containing 'x' (time series) and 'labels' columns.
        label_map: Dictionary mapping each letter to a number for prediction (should be a function that can be applied to a tensor).
        transform: Optional transform to be applied on the data.
        weight: Optional bool whether to weight by class or not.
        split_by_probes: Optional bool whether to split by probes or not.
        """
        self.data_columns = data_columns
        self.class_column = class_column
        self.transform = transform

        # Process each dataframe in the list of dfs
        self.x = []
        self.y = []
        self.weights = []

        # process splitting by probes and provide names for each file
        self.names = [df["file"] for df in dfs]

        for df in dfs:
            # Extract time series data and labels for each df
            x_tensor = torch.tensor(df[self.data_columns].values, dtype=torch.float32)
            y_tensor = torch.tensor(df[self.class_column].map(label_map).values, dtype=torch.long)
            self.x.append(x_tensor)
            self.y.append(y_tensor)

            # Handle weights if requested
            if weight:
                weights_tensor = torch.tensor(self.calculate_weights(df).tolist(), dtype=torch.float32)
            else:
                weights_tensor = torch.ones(x_tensor.shape, dtype=torch.float32)

            if ignore_N:
                mask = (df[self.class_column] == "N").values
                weights_tensor[mask] = 0.0

            self.weights.append(weights_tensor)

        # Pad sequences to the length of the longest time series in all datasets
        self.x = self.x
        self.y = self.y
        self.weights = self.weights

    def calculate_weights(self, df):
        '''
        calculate weights based on class makeup
        '''
        class_counts = df[self.class_column].value_counts().to_dict()
        total_samples = len(df)
        weights_by_class = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        return df[self.class_column].map(weights_by_class)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        weight = self.weights[idx]
        if self.transform:
            x = self.transform(x)
        return x, y, weight

def crop_after_upsampling(upsampled, downsampled, pad_left, pad_right):
    # Calculate the size difference and crop accordingly
    upsampled_size = upsampled.size(-1)
    downsampled_size = downsampled.size(-1)
    
    # Crop the tensor to match the size of the downsampled feature map
    if upsampled_size > downsampled_size:
        crop_left = pad_left
        crop_right = upsampled_size - downsampled_size - pad_right
        upsampled = upsampled[:, :, crop_left:upsampled_size - crop_right]
    
    return upsampled


def add_padding_for_downsampling(x, kernel_size, stride):
    # Calculate the necessary padding to keep the size even after downsampling
    pad_total = (stride - (x.size(-1) % stride)) % stride
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    # Apply padding
    x_padded = F.pad(x, (pad_left, pad_right))
    return x_padded, (pad_left, pad_right)  # Return padding values to track for cropping


def pad_or_crop(tensor, dim, target_size):
    # Calculate the difference in size
    current_size = tensor.size(dim)
    diff = target_size - current_size
    if diff > 0:
        # Padding is needed: apply symmetric padding on both sides along the target dimension
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * dim + 1)] = diff // 2       # pad before
        pad[-(2 * dim + 2)] = diff - diff // 2 # pad after
        tensor = F.pad(tensor, pad)
        
    elif diff < 0:
        # Cropping is needed: calculate the crop indices for symmetric cropping
        crop_start = (-diff) // 2
        crop_end = crop_start + target_size
        tensor = tensor.narrow(dim, crop_start, target_size)
        
    # If diff == 0, no operation is needed; the size is already correct
    return tensor


class UNetOutput:
    def __init__(self, scores):
        self.scores = scores
        self.num_classes = scores.size(-1)

    def nll(self, labels):
        return nn.CrossEntropyLoss()(self.scores.view(-1, self.num_classes), labels.view(-1))
    
    def marginals(self):
        return torch.softmax(self.scores, dim=-1)
    
    def argmax(self):
        return self.scores.argmax(dim=-1)
    
    def logits(self):
        return self.scores
    
class UNetCRFOutput(UNetOutput):
    def __init__(self, scores):
        super(UNetCRFOutput, self).__init__(scores)
        self.crf = LinearChainCRF(scores)

    def nll(self, labels):
        y_event = self.to_parts(labels, self.num_classes)
        return -self.crf.log_prob(y_event)
    
    def marginals(self):
        node_marginals = torch.zeros(self.scores.shape[0], self.scores.shape[1], self.num_classes, device=self.scores.device)
        edge_marginals = self.crf.marginals   # shape (batch, N-1, num_classes, num_classes)
        node_marginals[:, :-1, :] = edge_marginals.sum(-2)   # sum over next-state j
        node_marginals[:, 1:, :] += edge_marginals.sum(-1)   # sum over prev-state i
        node_marginals /= node_marginals.sum(-1, keepdim=True)  # normalize
        return node_marginals
    
    def argmax(self):
        return self.from_parts(self.crf.argmax)[0]
    
    def logits(self):
        return torch.log(self.marginals() + 1e-8)  # add small constant for numerical stability
    
    def to_parts(self, sequence: torch.Tensor, extra: int, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Convert a sequence representation to Markov edge indicators.

        Parameters
        ----------
        sequence : LongTensor of shape (B, N)
            Each entry in [0, C-1], where C = extra.
        extra : int
            Number of states (C).
        lengths : LongTensor of shape (B,), optional
            True sequence lengths for each batch element. If None, all are length N.

        Returns
        -------
        labels : LongTensor of shape (B, N-1, C, C)
            Markov edge indicators: labels[b, t, z_t, z_{t-1}] = 1 for valid transitions,
            0 elsewhere. Positions at or beyond lengths[b]-1 are zero.
        """
        C = extra
        device = sequence.device
        B, N = sequence.shape

        if lengths is None:
            lengths = sequence.new_full((B,), N, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        # Initialize all zeros on same device/dtype as sequence
        labels = sequence.new_zeros(B, N - 1, C, C)

        # Previous and next labels for all possible transitions
        prev_labels = sequence[:, :-1]   # (B, N-1)
        next_labels = sequence[:,  1:]   # (B, N-1)

        # Batch indices and time indices
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N - 1)   # (B, N-1)
        t_idx = torch.arange(N - 1, device=device).unsqueeze(0).expand(B, N - 1)  # (B, N-1)

        # Valid transitions are t < lengths[b] - 1
        valid_transitions = t_idx < (lengths - 1).unsqueeze(1)  # (B, N-1) bool

        # Flatten only the valid positions
        b_flat = b_idx[valid_transitions]
        t_flat = t_idx[valid_transitions]
        next_flat = next_labels[valid_transitions]
        prev_flat = prev_labels[valid_transitions]

        # Set indicators for valid edges
        labels[b_flat, t_flat, next_flat, prev_flat] = 1

        return labels.contiguous()
    
    def from_parts(self, edge: torch.Tensor):
        """
        Convert edges to sequence representation.

        Parameters
        ----------
        edge : Tensor of shape (B, N-1, C, C)
            Markov indicators: edge[b, t, z_t, z_{t-1}] = 1.

        Returns
        -------
        labels : LongTensor of shape (B, N)
            Reconstructed label sequences in [0, C-1].
        C : int
            Number of states.
        """
        B, N_1, C, _ = edge.shape
        N = N_1 + 1
        device = edge.device

        # edge[b, t, z_t, z_{t-1}] = 1
        # Sum over z_t → one-hot over previous state (z_{t-1})
        prev_onehot = edge.sum(dim=2)      # (B, N-1, C)
        # Sum over z_{t-1} → one-hot over next state (z_t)
        next_onehot = edge.sum(dim=3)      # (B, N-1, C)

        # Convert one-hot to label indices
        prev_labels = prev_onehot.argmax(dim=-1)  # (B, N-1)
        next_labels = next_onehot.argmax(dim=-1)  # (B, N-1)

        # Allocate output on same device, but long dtype
        labels = torch.zeros(B, N, device=device, dtype=torch.long)

        # At t = 0, label is previous state from first edge
        labels[:, 0] = prev_labels[:, 0]
        # For t >= 1, labels come from the "next" states of each edge
        labels[:, 1:] = next_labels

        return labels.contiguous(), C

class UNet1D(nn.Module):
    def __init__(self, input_size, 
                 output_size, 
                 growth_factor, 
                 features, 
                 num_layers, 
                 n_conv_steps_per_block, 
                 dropout_rate, 
                 block_kernel_size, 
                 up_down_sample_kernel_size, 
                 block_padding, 
                 bottleneck_type, 
                 transformer_window_size, 
                 embed_dim, 
                 transformer_layers, 
                 transformer_nhead,
                 block_type = "resnet",
                 upsample_type = "nearest",
                 downsample_type = "conv", 
                 crf_type = 'none',
                 skip_before_downsample = False,
                 norm_affine = False, 
                 skip_start_layer=0,
                 trial=False):
        super(UNet1D, self).__init__()
        self.num_layers = num_layers
        self.features = features
        self.growth_factor = growth_factor
        self.bottleneck_type = bottleneck_type
        self.transformer_window_size = transformer_window_size
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.transformer_nhead = transformer_nhead
        self.crf_type = crf_type
        self.skip_start_layer = skip_start_layer
        self.block_type = block_type
        self.upsample_type = upsample_type
        self.downsample_type = downsample_type
        self.skip_before_downsample = skip_before_downsample
        self.norm_affine = norm_affine

        if trial:
            # integers from a power-of-two grid
            self.epochs      = 96 #trial.suggest_int("epochs", 64, 128, step=16)
            self.num_layers  = trial.suggest_int("num_layers", 6, 8, step=1)
            self.n_conv_steps_per_block = 2 # trial.suggest_int("n_conv_steps_per_block", 1, 3, step=1)
            self.features    = 64 #trial.suggest_int("features", 32, 128, step=32)
            self.lr          = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
            use_dropout0     = False # trial.suggest_categorical("dropout_is_zero", [True, False])
            self.dropout_rate = 0.0 if use_dropout0 else trial.suggest_float("dropout_pos", 1e-2, 0.5, log=True)
            use_wd0          = False # trial.suggest_categorical("weight_decay_is_zero", [True, False])
            self.weight_decay = 0.0 if use_wd0 else trial.suggest_float("weight_decay_pos", 1e-8, 1e-3, log=True)
            self.crf_type = trial.suggest_categorical("crf_type", ['crf', 'shared_transition_crf'])
            self.skip_start_layer = trial.suggest_int("skip_start_layer", 0, 4, step=1)
            self.block_type = 'resnet' # trial.suggest_categorical("block_type", ['resnet', 'simple'])
            self.upsample_type = 'nearest' # trial.suggest_categorical("upsample_type", ['convtranspose', 'nearest', 'linear'])
            self.downsample_type = 'maxpool' # trial.suggest_categorical("downsample_type", ['conv', 'avgpool', 'maxpool'])
            self.growth_factor = 1 # trial.suggest_categorical("growth_factor", [1, 2])
            self.skip_before_downsample = True # trial.suggest_categorical("skip_before_downsample", [True, False])
            self.norm_affine = True # trial.suggest_categorical("norm_affine", [True, False])

            if self.bottleneck_type == "attention" or self.bottleneck_type == "windowed_attention":
                self.transformer_window_size = -1 # trial.suggest_int("transformer_window_size", 100, 400, step=100)
                self.embed_dim = self.features  # tie to features
                self.transformer_layers = trial.suggest_int("transformer_layers", 1, 3, step=1)
                self.transformer_nhead = trial.suggest_categorical("heads_per_channel", [4, 8, 16])
            else:
                self.bottleneck_type = "block"

        # input layer
        self.in_conv = nn.Conv1d(in_channels=input_size, out_channels=features, kernel_size=1)

        # Encoding layers
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            encoder = EncoderBlock( 
                n_conv_steps_per_block,
                in_channels=features,
                out_channels=growth_factor*features,
                kernel_size=block_kernel_size,
                stride=2,
                dilation=1,
                dropout=dropout_rate,
                block_type=self.block_type,
                downsample_type=self.downsample_type,
                norm_affine=self.norm_affine,
                skip_before_downsample=self.skip_before_downsample
            )
            self.encoders.append(encoder)
            features *= growth_factor  # Increase feature size

        # Bottleneck
        if self.bottleneck_type == "block" or self.transformer_layers == 0:
            # in the bottleneck, we don't do any growth and just map features-->features
            
            self.bottleneck = EncoderBlock( 
                n_conv_steps_per_block,
                in_channels=features,
                out_channels=features,
                kernel_size=block_kernel_size,
                stride=1,
                dilation=1,
                dropout=dropout_rate,
                block_type=self.block_type,
                downsample_type=self.downsample_type,
                norm_affine=self.norm_affine
            )
        elif self.bottleneck_type == "attention":
            self.to_embed = nn.Conv1d(features, embed_dim, 1)
            self.from_embed = nn.Conv1d(embed_dim, features, 1)
            self.transfomer = TransformerBotleneck(self.transformer_layers, self.embed_dim,
                    nhead=self.transformer_nhead,
                    window_size=self.transformer_window_size,
                    dim_feedforward=-1,
                    dropout=dropout_rate,
                    activation="gelu")
            self.bottleneck = nn.Sequential(self.to_embed,
                                            self.transfomer,
                                            self.from_embed)
        else:
            assert False
        

        # Decoding layers
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            decoder = DecoderBlock( 
                n_conv_steps_per_block,
                in_channels=features,
                out_channels=features // growth_factor,
                kernel_size=block_kernel_size,
                stride=2,
                dilation=1,
                dropout=0.0,
                block_type=self.block_type,
                upsample_type=self.upsample_type,
                norm_affine=self.norm_affine,
                skip_before_downsample=self.skip_before_downsample
            )
            self.decoders.append(decoder)
            features //= growth_factor  # Decrease feature size

        # output layer
        if self.crf_type != 'crf':
            self.out_conv = nn.Conv1d(in_channels=features, out_channels=output_size, kernel_size=1)
        else:
            self.out_conv = nn.Conv1d(in_channels=features, out_channels=output_size * output_size, kernel_size=1)
        self.transition = nn.Parameter(torch.randn(output_size, output_size))
        self.init = nn.Parameter(torch.randn(output_size))
        

    def forward(self, x):
        # Encoding path
        encodings = []
        x = self.in_conv(x)

        initial_size = x.shape[2]

        for i in range(self.num_layers):
            x, skip = self.encoders[i](x, return_skip=True)
            if i < self.skip_start_layer:
                skip = torch.zeros_like(skip)
            # if growth factor is not 1, then downconv
            encodings.append(skip)
            
        # Bottleneck
        residual = x
        x = self.bottleneck(x)
        x = x + pad_or_crop(residual, dim=2, target_size=x.shape[2])  # Residual connection

        # Decoding path
        for i in range(self.num_layers):
            prev_encoding = encodings[-(i+1)]
            x = self.decoders[i](x, prev_encoding)

        # pad up to the inital size
        x = pad_or_crop(x, dim=2, target_size=initial_size)
        x = self.out_conv(x)
        if self.crf_type == 'shared_transition_crf':
            scores = x.permute(0,2,1) # shape (batch, seq_len, num_classes, 1)
            edge = self.transition[None, None, :, :] + scores[:, 1:, :, None]
            prev_term = (self.init[None, None, :] + scores[:, 0:1, :])  # (B, 1, C)
            edge[:, 0, :, :] = edge[:, 0, :, :] + prev_term[:, 0:1, :]      # broadcast over current state
                
            # shape (batch, seq_len-1, num_classes, num_classes)
            scores = edge
        elif self.crf_type == 'crf':
            batch_size, seq_len, _ = x.permute(0,2,1).shape
            scores = x.permute(0,2,1).contiguous().reshape(batch_size, seq_len, self.transition.shape[0], self.transition.shape[1])
            edge = scores[:, 1:, :, :] + self.transition[None, None, :, :] # shape (batch, seq_len-1, num_classes, num_classes)
            edge[:, 0, :, :] = edge[:, 0, :, :] + torch.logsumexp(scores[:, 0, :, :], dim=2, keepdim=False)[:, None, :] + self.init[None, None, :]  # add initial term to first edge
            scores = edge
        else:
            scores = x.permute(0,2,1) # shape (batch, seq_len, num_classes)

        scores = scores.contiguous()

        if self.crf_type == 'none':
            return UNetOutput(scores)
        else:
            return UNetCRFOutput(scores)
    

def make_activation(activation: str):
    act = activation.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif act == "gelu":
        return nn.GELU()
    elif act in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    elif act == "tanh":
        return nn.Tanh()
    elif act in ("identity", "none"):
        return nn.Identity()
    else:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            "Choose from: relu, leaky_relu, gelu, silu/swish, tanh, identity/none."
        )
    
def make_gn(num_channels: int, max_groups: int = 8, eps: float = 1e-5, affine: bool = False):
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    #   return nn.GroupNorm(g, num_channels, eps=eps, affine=affine) 
    return nn.GroupNorm(num_channels, num_channels, eps=eps, affine=affine)

class EncoderBlock(nn.Module):
    def __init__(self, 
        layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_affine: bool = True,
        eps: float = 1e-5,
        downsample_type: str = "conv",
        downsample_kernel_size: int = 3,
        block_type: str = "resnet",
        skip_before_downsample: bool = False,
    ):
        """
        Defines a single encoding block consisting of nX: Conv1d, GroupNorm, and GELU.
        """
        super(EncoderBlock, self).__init__()
        self.skip_before_downsample = skip_before_downsample
        self.block_type = {'resnet': ResNet1DBlock, 'simple': Simple1DBlock}[block_type]
        self.block = nn.Sequential()
        for _ in range(layers):
            self.block.append(self.block_type(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                groups=groups,
                dropout=dropout,
                activation=activation,
                norm_affine=norm_affine,
                eps=eps,
            ))

        
        padding = ((downsample_kernel_size - 1) // 2)
        if downsample_type == "conv":
            self.output = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=downsample_kernel_size,
                stride=stride, padding=padding),
                make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine),
                make_activation(activation),
            )
        elif downsample_type == "avgpool":
            self.output = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            )
        elif downsample_type == "maxpool":
            self.output = nn.Sequential(
                nn.MaxPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown downsample_type '{downsample_type}'. Choose from: conv, avgpool, maxpool.")
            
    def forward(self, x, return_skip=False):
        x = self.block(x) 
        padded = x #torch.nn.functional.pad(x, (0, x.shape[2] % 2))  # Pad to even length if necessary
        output = self.output(padded)
        if return_skip:
            return output, (x if self.skip_before_downsample else output)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, 
        layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_affine: bool = True,
        eps: float = 1e-5,
        upsample_type: str = "nearest",
        upsample_kernel_size: int = 3,
        block_type: str = "resnet",
        skip_before_downsample: bool = False,
    ):
        """
        Defines a single decoding block consisting of nX: Conv1d, GroupNorm, and GELU.
        """
        super(DecoderBlock, self).__init__()
        self.skip_before_downsample = skip_before_downsample
        self.block_type = {'resnet': ResNet1DBlock, 'simple': Simple1DBlock}[block_type]
        self.block = nn.Sequential()

        padding = ((upsample_kernel_size - 1) // 2)
        if upsample_type == "nearest":
            self.input = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest'),
                nn.Conv1d(in_channels, out_channels, kernel_size=upsample_kernel_size, padding=padding),
                make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine),
                make_activation(activation),
            )
        elif upsample_type == "linear":
            self.input = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=upsample_kernel_size, padding=padding),
                make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine),
                make_activation(activation),
            )
        elif upsample_type == "convtranspose":
            self.input = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upsample_kernel_size, stride=stride, padding=padding, output_padding=stride-1),
                make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine),
                make_activation(activation),
            )
        else:
            raise ValueError(f"Unknown upsample_type '{upsample_type}'. Choose from: nearest, linear, convtranspose.")
        
        in_channels = out_channels  # Update in_channels for the next layer
        for _ in range(layers):
            self.block.append(self.block_type(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                groups=groups,
                dropout=dropout,
                activation=activation,
                norm_affine=norm_affine,
                eps=eps,
            ))
            
    def forward(self, x, skip):
        if self.skip_before_downsample:
            x = pad_or_crop(self.input(x), dim=2, target_size=skip.shape[2]) + skip
        else:
            x = self.input(pad_or_crop(x, dim=2, target_size=skip.shape[2]) + skip)
        return self.block(x)

class ResNet1DBlock(nn.Module):
    """
    Basic 1D ResNet block using InstanceNorm1d:
      Conv-IN-Act-Conv-IN + skip connection.

    Input/Output: (B, C, T)
    - If in_channels != out_channels or stride != 1, uses a 1x1 conv projection on the skip.
    - InstanceNorm1d uses affine=True by default here (learnable scale/shift).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Use an odd kernel_size to preserve length with 'same' padding."
        padding = ((kernel_size - 1) // 2) * dilation

        # --- activation (inlined) ---
        self.act = make_activation(activation)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # --- main path ---
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.in1 = make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.in2 = make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine)

        # --- skip path ---
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine),
            )
        else:
            self.proj = nn.Identity()

        # Optional: start residual branch near-zero (only if affine=True)
        if norm_affine and self.in2.weight is not None:
            nn.init.zeros_(self.in2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.in1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.in2(out)

        out = out + identity
        out = self.act(out)
        return out
    
class Simple1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Use an odd kernel_size to preserve length with 'same' padding."
        assert stride == 1, "ResidualConvBlock does not support downsampling. Use stride=1 and add downsampling separately if needed."
        assert in_channels == out_channels, "ResidualConvBlock requires in_channels == out_channels for the skip connection."
        padding = ((kernel_size - 1) // 2) * dilation

        # --- activation (inlined) ---
        self.act = make_activation(activation)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # --- main path ---
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.in1 = make_gn(out_channels, max_groups=8, eps=eps, affine=norm_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.in1(out)
        out = self.act(out)
        out = self.drop(out)
        out = out + x
        return out


try:
    import xformers.ops as xops
    _HAS_XFORMERS = True
except Exception:
    _HAS_XFORMERS = False

class WindowedMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=-1, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, is_causal=False):
        # x: (B, T, C)
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        if _HAS_XFORMERS:
            if self.window_size <= 0:
                if is_causal:
                    attn_bias = xops.fmha.attn_bias.LowerTriangularMask()
                else:
                    attn_bias = None
            else:
                if is_causal:
                    attn_bias = xops.fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(self.window_size)
                else:
                    attn_bias = xops.fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=self.window_size, window_right=self.window_size)
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)
        else:
            # Pure PyTorch dense masked window attention
            q_ = q.transpose(1, 2)  # (B,H,T,D)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            out_ = F.scaled_dot_product_attention(
                q_, k_, v_,
                dropout_p=self.dropout,
                is_causal=is_causal
            )
            out = out_.transpose(1, 2)  # (B,T,H,D)

        out = out.reshape(B, T, C)
        return self.out_proj(out)

class WindowedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer using WindowedMHA for (local) self-attention.

    Input/Output:
      x: (B, T, C) -> (B, T, C)

    Notes:
      - This version assumes you don't need a key_padding_mask. If you do, you'll
        want to extend WindowedMHA to accept it (easy to add to the dense SDPA path).
      - norm_first=True gives Pre-LN (usually preferred for stability).
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_first = norm_first

        # --- self-attention ---
        self.self_attn = WindowedMHA(
            embed_dim=d_model,
            num_heads=nhead,
            window_size=window_size,
            dropout=dropout,
            bias=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # --- feedforward ---
        if activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() in ("silu", "swish"):
            act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = act
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def _sa_block(self, x: torch.Tensor, is_causal: bool) -> torch.Tensor:
        # WindowedMHA returns (B,T,C)
        attn_out = self.self_attn(x, is_causal=is_causal)
        return self.dropout1(attn_out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_ff(x)
        x = self.linear2(x)
        return self.dropout2(x)

    def forward(self, x: torch.Tensor, *, is_causal: bool = False) -> torch.Tensor:
        if self.norm_first:
            # Pre-LN
            x = x + self._sa_block(self.norm1(x), is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
            return x
        else:
            # Post-LN (closer to original Transformer)
            x = self.norm1(x + self._sa_block(x, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
            return x

    
class TransformerBotleneck(nn.Module):
    def __init__(self, transformer_layers: int, embed_dim: int,
        nhead: int,
        window_size: int,
        dim_feedforward: int = -1,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,):

        super(TransformerBotleneck, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.transformer_layers = transformer_layers
        if dim_feedforward == -1:
            dim_feedforward = 2 * embed_dim

        self.positional_encoder = PositionalEncoding1D(self.embed_dim)
        self.layers = nn.ModuleList([
            WindowedTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                window_size=window_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(self.transformer_layers)
        ])
        self.transformer_encoder = nn.Sequential(*self.layers)

    def forward(self, x):
        # Apply a positional embedding
        x = x.permute(0, 2, 1) # batch, seq_len, channels
        x_positional_encoded = x + self.positional_encoder(x)
        # Transformer Encoding
        encoded = self.transformer_encoder(x_positional_encoded)
        encoded = encoded.permute(0, 2, 1) # batch, channels, seq_len
        return encoded
    
        # self.batch_size = batch_size
        # self.epochs=epochs
        # self.lr=lr
        # self.weight_decay = weight_decay
        # self.dropout_rate = dropout_rate
        # self.block_kernel_size = block_kernel_size
        # self.up_down_sample_kernel_size = up_down_sample_kernel_size
        # self.block_padding = block_padding
        # self.n_conv_steps_per_block = n_conv_steps_per_block
        # self.num_layers = num_layers
        # self.growth_factor = growth_factor
        # self.features = features
        # self.bottleneck_type = bottleneck_type
        # self.transformer_window_size = transformer_window_size
        # self.loss_gamma = loss_gamma
        # self.loss_alpha = loss_alpha (neither of these seem to be used)
        # self.embed_dim = embed_dim
        # self.transformer_nhead = transformer_nhead
        # self.crf_type = crf_type
        # self.skip_start_layer = skip_start_layer
        # self.block_type = block_type
        # self.upsample_type = upsample_type
        # self.downsample_type = downsample_type
        # self.skip_before_downsample = skip_before_downsample
        # self.norm_affine = norm_affine
        
        # if self.embed_dim is None:
        #     self.embed_dim = self.features * (self.growth_factor**self.num_layers)

        # self.transformer_layers = transformer_layers

        # if self.transformer_nhead is None:
        #     self.transformer_nhead = 4

        # if trial:
        #     # integers from a power-of-two grid
        #     self.epochs      = 96 #trial.suggest_int("epochs", 64, 128, step=16)
        #     self.num_layers  = trial.suggest_int("num_layers", 6, 8, step=1)
        #     self.n_conv_steps_per_block = 2 # trial.suggest_int("n_conv_steps_per_block", 1, 3, step=1)
        #     self.features    = 64 #trial.suggest_int("features", 32, 128, step=32)
        #     self.lr          = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        #     use_dropout0     = False # trial.suggest_categorical("dropout_is_zero", [True, False])
        #     self.dropout_rate = 0.0 if use_dropout0 else trial.suggest_float("dropout_pos", 1e-2, 0.5, log=True)
        #     use_wd0          = False # trial.suggest_categorical("weight_decay_is_zero", [True, False])
        #     self.weight_decay = 0.0 if use_wd0 else trial.suggest_float("weight_decay_pos", 1e-8, 1e-3, log=True)
        #     self.crf_type = trial.suggest_categorical("crf_type", ['crf', 'shared_transition_crf'])
        #     self.skip_start_layer = trial.suggest_int("skip_start_layer", 0, 4, step=1)
        #     self.block_type = 'resnet' # trial.suggest_categorical("block_type", ['resnet', 'simple'])
        #     self.upsample_type = 'nearest' # trial.suggest_categorical("upsample_type", ['convtranspose', 'nearest', 'linear'])
        #     self.downsample_type = 'maxpool' # trial.suggest_categorical("downsample_type", ['conv', 'avgpool', 'maxpool'])
        #     self.growth_factor = 1 # trial.suggest_categorical("growth_factor", [1, 2])
        #     self.skip_before_downsample = True # trial.suggest_categorical("skip_before_downsample", [True, False])
        #     self.norm_affine = True # trial.suggest_categorical("norm_affine", [True, False])

        #     if self.bottleneck_type == "attention" or self.bottleneck_type == "windowed_attention":
        #         self.transformer_window_size = -1 # trial.suggest_int("transformer_window_size", 100, 400, step=100)
        #         self.embed_dim = self.features  # tie to features
        #         self.transformer_layers = trial.suggest_int("transformer_layers", 1, 3, step=1)
        #         self.transformer_nhead = trial.suggest_categorical("heads_per_channel", [4, 8, 16])
        #     else:
        #         self.bottleneck_type = "block"