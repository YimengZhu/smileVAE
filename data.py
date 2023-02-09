from torch.utils.data import Dataset

SMILE = 3
DENSITY = 6
CALORICITY = 7
MELTING = 8


class Monecular(Dataset):
    def __init__(self, datapath):
        with open(datapath) as fd:
            self.samples = fd.readlines()[1:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].strip()
        sample_columns = sample.split('\t')

        feature = sample_columns[SMILE]
        
        density = sample_columns[DENSITY]
        caloricity = sample_columns[CALORICITY]
        melting = sample_columns[MELTING]
        
        return feature, (density, caloricity, melting)

