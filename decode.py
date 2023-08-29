# coding=utf-8
import torch
from data import Monecular
from model import make_model
from torch.utils.data import DataLoader

model = make_model('VAE')
checkpoint = torch.load('./model.pth')
model.load_state_dict(checkpoint)

dataset = Monecular('145sample_data.csv')
test_loader = DataLoader(dataset)
idx2char = {value: key for key, value in dataset.char2idx.items()}

model.eval()
for feature, label_word in test_loader:
    reconstruct_list, properties_list = model(feature)
    for reconstruct, properties in zip(reconstruct_list, properties_list):
        feature = feature.squeeze()
        reconstruct = reconstruct.squeeze()

        pred = torch.argmax(reconstruct, dim=1)
        pred = pred.tolist()
        pred_word = []

        for v in pred:
            pred_word.append(idx2char[v])

        print(label_word[0], '--->', ''.join(pred_word))