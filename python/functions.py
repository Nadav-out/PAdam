import torch

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
def calculate_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    num_batches = 0

    for images, _ in dataloader:
        num_batches += 1
        for i in range(3):
            mean_sum[i] += images[:,i,:,:].mean()
            std_sum[i] += images[:,i,:,:].std()

    mean = mean_sum / num_batches
    std = std_sum / num_batches

    return mean, std

def fraction_small_weights(param, threshold):
        small_weights = torch.sum(torch.abs(param.data) < threshold).item()
        total_weights = param.nelement()
        return small_weights / total_weights
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


