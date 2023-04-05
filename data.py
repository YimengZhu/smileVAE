import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

SMILE = 3
DENSITY = 5
CALORICITY = 6
MELTING = 7


class Monecular(Dataset):
    def __init__(self, datapath):
        with open(datapath) as fd:
            self.samples = fd.readlines()[1:]
            self.vocab = set()
            smile_words = [sample.split(',')[SMILE] for sample in self.samples]
            for w in smile_words:
                for c in w:
                    self.vocab.add(c)
            
            self.vocab = list(self.vocab)
            self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].strip()
        sample = sample.split(',')

        char_list = [self.char2idx[char] for char in sample[SMILE]]
        feature = torch.tensor(char_list)
        feature = F.one_hot(feature, num_classes=len(self.vocab))
        
        density = torch.tensor(sample[DENSITY])
        caloricity = torch.tensor(sample[CALORICITY])
        melting = torch.tensor(sample[MELTING])
        
        return feature, (density, caloricity, melting)

