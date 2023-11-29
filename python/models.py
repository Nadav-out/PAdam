import torch
import torch.nn as nn
import torch.nn.functional as F
from Optimizers import *
import inspect


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






    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        

        
        
        
def conv_block(in_channels, out_channels, activation=False, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels)]
    if activation: layers.append(nn.ReLU(inplace=True))
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),nn.ReLU(inplace=True))

        self.res1 = nn.Sequential(conv_block(64, 64,activation=True), conv_block(64, 64))
        self.res2 = nn.Sequential(conv_block(64, 64,activation=True), conv_block(64, 64))
        
        self.downsample1=nn.Sequential(conv_block(64, 128,pool=True)) 
        self.res3 = nn.Sequential(conv_block(64, 128,activation=True, pool=True),
                                  conv_block(128,128))
        self.res4 = nn.Sequential(conv_block(128, 128,activation=True), conv_block(128, 128,activation=True))
        
        self.res5 = nn.Sequential(conv_block(128, 256,activation=True, pool=True),conv_block(256,256))
        self.downsample2 = nn.Sequential(conv_block(128, 256,pool=True,activation=True))
        self.res6 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256,activation=True))
        
        self.res7 = nn.Sequential(conv_block(256, 512,activation=True, pool=True),
                                   conv_block(512,512,activation=True))
        self.downsample3 = nn.Sequential(conv_block(256,512,activation=True,pool=True))
        self.res8 = nn.Sequential(conv_block(512, 512,activation=True), conv_block(512, 512,activation=True))

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.1),
                                        nn.Linear(512, num_classes))
        
        self.apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

            
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + self.downsample1(out)
        out = self.res4(out) + out
        out = self.res5(out) + self.downsample2(out)
        out = self.res6(out) + out
        out = self.downsample3(out) +self.res7(out)
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
        
        print(f'WD={weight_decay}')
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
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
        
        print(f"using fused AdamW: {use_fused}")

        return optimizer

