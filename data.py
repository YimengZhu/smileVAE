import torch
from torch.utils.data import Dataset

SMILE = 3
DENSITY = 5
CALORICITY = 6
MELTING = 7


class Monecular(Dataset):
    def __init__(self, datapath):
        with open(datapath) as fd:
            self.samples = fd.readlines()[1:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].strip()
        sample = sample.split(',')

        feature = torch.tensor(sample[SMILE])
        
        density = torch.tensor(sample[DENSITY])
        caloricity = torch.tensor(sample[CALORICITY])
        melting = torch.tensor(sample[MELTING])
        
        return feature, (density, caloricity, melting)

