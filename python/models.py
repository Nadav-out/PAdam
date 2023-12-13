from ast import Raise
import torch
import torch.nn as nn
import torch.nn.functional as F
from Optimizers import *
import inspect
import pandas as pd



class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)





class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, stride=2)
        
        # reshape
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        # Dropout
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

        
        
def conv_block(in_channels, out_channels, kernel_size=3, padding=1, activation=False, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False), 
              nn.BatchNorm2d(out_channels)]
    if activation: layers.append(nn.ReLU(inplace=True))
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64, kernel_size=7, padding=4, activation=True)
        

        self.res1 = nn.Sequential(
            conv_block(64, 64,activation=True), 
            conv_block(64, 64)
            )
        
        self.res2 = nn.Sequential(
            conv_block(64, 64,activation=True), 
            conv_block(64, 64)
            )
        
        self.downsample1=conv_block(64, 128,pool=True)
        
        self.res3 = nn.Sequential(
            conv_block(64, 128,activation=True, pool=True),
            conv_block(128,128)
            )
        
        self.res4 = nn.Sequential(
            conv_block(128, 128,activation=True), 
            conv_block(128, 128,activation=True)
            )
        
        self.res5 = nn.Sequential(
            conv_block(128, 256,activation=True, pool=True),
            conv_block(256,256)
            )
        
        self.downsample2 = conv_block(128, 256,pool=True,activation=True)
        
        self.res6 = nn.Sequential(
            conv_block(256, 256,activation=True), 
            conv_block(256, 256,activation=True)
            )
        
        self.res7 = nn.Sequential(
            conv_block(256, 512,activation=True, pool=True),
            conv_block(512,512,activation=True))
        
        self.downsample3 = conv_block(256,512,activation=True,pool=True)
        
        self.res8 = nn.Sequential(
            conv_block(512, 512,activation=True), 
            conv_block(512, 512,activation=True)
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten(), 
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
            )
        
        
        # Initialize weights
        self.apply(self.init_weights)
        
        # Split in to decayed vs non-decayed parameters
        self._initialize_param_groups()  
        
        # initialized decayed parameters df to track sparsity
        self._initialize_sparsity_df()
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + self.downsample1(out)
        out = self.res4(out) + out
        out = self.res5(out) + self.downsample2(out)
        out = self.res6(out) + out
        out = self.res7(out) + self.downsample3(out)
        out = self.res8(out) + out
        out = self.classifier(out)
        return out

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    
    
   
    def _initialize_param_groups(self):
        '''
        Initializes and splits model parameters into decayed and non-decayed groups based on their tensor dimension.
        The logic is that we don't want to decay 1D tensors (e.g., biases, batch normalization parameters).
        For analysis purposes, the layer name, sublayer name, and type of each parameter are also stored.
        This method populates the lists self.decay_params and self.nodecay_params with these details.
        '''
        self.decay_params = []
        self.nodecay_params = []

        for name, module in self.named_modules():
            if hasattr(module, 'parameters'):
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        if param.dim() >= 2:
                            self.decay_params.append((name, param, type(module).__name__))
                        else:
                            self.nodecay_params.append((name, param, type(module).__name__))
        self.num_decay_params = sum(p[1].numel() for p in self.decay_params)
        num_nodecay_params = sum(p[1].numel() for p in self.nodecay_params)
        print(f"Number of trainable parameters: {self.num_decay_params+num_nodecay_params:,}, grouped into:")
        print(f"{len(self.decay_params)} decayed parameter tensors, with {self.num_decay_params:,} parameters")
        print(f"{len(self.nodecay_params)} non-decayed parameter tensors, with {num_nodecay_params:,} parameters")
     
    def _initialize_sparsity_df(self):
        '''
        Initializes a pandas DataFrame to monitor the evolution of parameter sparsity during training. 
        Each row in the DataFrame corresponds to a parameter group as defined by the _initialize_param_groups method. 
        The columns 'Layer', 'Sublayer', 'Component', and 'paramsCount' capture the layer name, sublayer name, 
        type of the layer, and the count of parameters in that layer, respectively.
        '''
        self.sparsity_df = pd.DataFrame(columns=['Layer', 'Sublayer', 'Component', 'paramsCount'])
        new_rows = []

        for name, param, component in self.decay_params:
            parts = name.split('.')
            layer, sublayer = (parts[0], '.'.join(parts[1:])) if len(parts) > 1 else (parts[0], '')

            params_count = param.numel()

            new_row = {
                'Layer': layer,
                'Sublayer': sublayer,
                'Component': component,
                'paramsCount': params_count
            }
            new_rows.append(new_row)

        self.sparsity_df = pd.concat([self.sparsity_df, pd.DataFrame(new_rows)], ignore_index=True)

        
    

    def append_small_weight_vec(self, threshold, epoch):
        '''
        Appends the sparsity DataFrame with the count of small weights for a given epoch.
        Small weights are defined as those whose absolute value is less than the specified threshold.
        The method calculates the count of such weights for each parameter group in decay_params and 
        appends the DataFrame with this information. It also returns the current overall sparsity.
        '''
        small_weight_counts = []
        column_name = f'SmallWeights_epoch{epoch}'

        for param in self.decay_params:
            small_weight_count = (param[1].abs() < threshold).sum().item()
            small_weight_counts.append(small_weight_count)

        # Append to the sparsity DataFrame
        if column_name not in self.sparsity_df.columns:
            self.sparsity_df[column_name] = small_weight_counts
        else:
            self.sparsity_df[column_name].update(pd.Series(small_weight_counts))

        # Calculate and return the current sparsity
        cur_sparsity = self.sparsity_df[column_name].sum() / self.num_decay_params
        return cur_sparsity
    
    
    def decayed_weights_histogram(self):
        # Aggregate all decayed weights
        decayed_weights = torch.cat([param.data.view(-1) for name, param, _ in self.decay_params if param.requires_grad]).cpu()
        counts, bins=(torch.abs(decayed_weights) + 1e-13).log10().histogram(bins=50,range=(-13, 0))
        log_counts = counts.log10()
        log_counts[log_counts == float('-inf')] = -1


        # Apply log10 transformation
        return (counts.numpy(), bins.numpy())

        

    def configure_optimizers(self, optimizer_name, weight_decay, learning_rate, p_norm, betas, device_type):
        # Create AdamW/PAdam optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_name == 'AdamW':
            optim_groups = [
                {'params': [p[1] for p in self.decay_params], 'weight_decay': weight_decay},
                {'params': [p[1] for p in self.nodecay_params], 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        elif optimizer_name == 'PAdam':
            optim_groups = [
                {'params': [p[1] for p in self.decay_params], 'lambda_p': weight_decay},
                {'params': [p[1] for p in self.nodecay_params], 'lambda_p': 0.0}
            ]
            optimizer = PAdam(optim_groups, lr=learning_rate, p_norm=p_norm, betas=betas, **extra_args)
        
        elif optimizer_name == 'Adam_L1':
            optim_groups = [
                {'params': [p[1] for p in self.decay_params], 'l1_lambda': weight_decay},
                {'params': [p[1] for p in self.nodecay_params], 'l1_lambda': 0.0}
            ]
            optimizer = Adam_L1(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        else:
            raise ValueError("optimizer_name must be 'AdamW', 'PAdam' or 'Adam_L1'")

        print(f"using fused Adam: {use_fused}")

        return optimizer



