import argparse
import torch 
from torch.utils.data import DataLoader
from data import Monecular
from model import make_model


def train(model, data_loader, optimzizer, epochs):
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i, (feature, _) in enumerate(train_loader):
            prediction = model(feature)
            loss = criterion(feature, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'model.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['VAE', 'AE'], default='VAE')
    parser.add_argument('--smile_length', type=int, default=0)
    args = parser.parse_args()

    dataset = Monecular('sample_data.csv', max_length=args.smile_length)
    train_loader = DataLoader(dataset)

    model = make_model(args.model_type)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

    epochs = 5

    train(model, train_loader, optimizer, epochs)