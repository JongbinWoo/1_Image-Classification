from torch.utils.data import DataLoader

def get_loader(dataset, batch_size, num_workers, sampler):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)