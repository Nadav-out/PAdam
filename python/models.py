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
        
        
        # Store the number of parameters for each layer
        self.apply(self.init_weights)
        # Initialize and count parameters
        self.layer_param_count, self.layer_regulated_param_count = self.count_parameters()
        self.sparsity_df = pd.DataFrame(columns=['Layer', 'Sublayer', 'Component', 'paramsCount'])
        self._initialize_sparsity_df()


    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    
    def _initialize_sparsity_df(self):
        new_rows = []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parts = name.split('.')
                layer, sublayer = (parts[0], '.'.join(parts[1:])) if len(parts) > 1 else (parts[0], '')
                component = type(module).__name__

                # Calculate total number of parameters for this module
                params_count = sum(p.numel() for p in module.parameters())

                new_row = {
                    'Layer': layer,
                    'Sublayer': sublayer,
                    'Component': component,
                    'paramsCount': params_count
                }
                new_rows.append(new_row)

        self.sparsity_df = pd.concat([self.sparsity_df, pd.DataFrame(new_rows)], ignore_index=True)

    def get_small_weight_vec(self, threshold):
        small_weight_counts = []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                small_weight_count = sum((p.abs() < threshold).sum().item() for p in module.parameters())
                small_weight_counts.append(small_weight_count)

        return small_weight_counts




            
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
    
    def configure_optimizers(self, optimizer_name, weight_decay, learning_rate, p_norm, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW/PAdam optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        if optimizer_name == 'AdamW':
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
                ]
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        elif optimizer_name == 'PAdam':
            optim_groups = [
                {'params': decay_params, 'lambda_p': weight_decay},
                {'params': nodecay_params, 'lambda_p': 0.0}
                ]
            optimizer = PAdam(optim_groups, lr=learning_rate,p_norm=p_norm, betas=betas, **extra_args)
        
        print(f"using fused Adam: {use_fused}")

        return optimizer
    
    def count_parameters(self):
        layer_param_count = {}
        layer_regulated_param_count = {}  # For regulated (weight decay) parameters

        for name, param in self.named_parameters():
            if param.requires_grad:
                # Extract module name from parameter name
                module_name = name.split('.')[0]
                
                # Count all trainable parameters
                layer_param_count[module_name] = layer_param_count.get(module_name, 0) + param.numel()

                # Count regulated parameters (2D or higher)
                if param.dim() >= 2:
                    layer_regulated_param_count[module_name] = layer_regulated_param_count.get(module_name, 0) + param.numel()

        total_params = sum(layer_param_count.values())
        print(f"Number of trainable parameters: {total_params:,}")

        return layer_param_count, layer_regulated_param_count

    
    def count_small_weights(self, threshold=1e-13):
        small_weights_count = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() >= 2:  # Regularized weights
                # Extract the higher-level module name from the parameter name
                module_name = name.split('.')[0]
                # Count the small weights in this parameter
                count = (param.abs() < threshold).sum().item()
                if module_name in small_weights_count:
                    small_weights_count[module_name] += count
                else:
                    small_weights_count[module_name] = count
        return small_weights_count
    
    

    def _create_sparsity_dict_structure(self, parent_name='', module=None):
        if module is None:
            module = self
        sparsity_dict = {}
        for name, layer in module.named_children():
            layer_identifier = f'{parent_name}.{name}' if parent_name else name

            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                if parent_name:
                    if parent_name not in sparsity_dict:
                        sparsity_dict[parent_name] = {}
                    sparsity_dict[parent_name][name] = 0
                else:
                    sparsity_dict[name] = 0
            else:
                sub_dict = self._create_sparsity_dict_structure(layer_identifier, layer)
                if sub_dict:
                    sparsity_dict[layer_identifier] = sub_dict
        return sparsity_dict




