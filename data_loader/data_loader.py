from torch.utils.data import DataLoader

def get_loader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)