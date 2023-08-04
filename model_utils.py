import torch
import torch.nn as nn
from torch_lr_finder import LRFinder


def get_incorrrect_predictions(model, loader, device):
    """ Gets the incorrect prections """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect


def plot_lr(model, optimizer, train_loader):
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device='mps')
    lr_finder.reset()
    lr_finder.range_test(train_loader, start_lr=1e-4, end_lr=50, num_iter=100, step_mode="exp")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
