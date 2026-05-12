from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn

def f1_score(labels, preds, output_transform):
    preds = output_transform(preds)
    # Go from probabilities to number of cards
    preds = 1.49 * preds / torch.max(preds)
    labels = labels / torch.min(labels[labels>0])
    zero = torch.zeros_like(preds)
    TP = torch.sum(torch.maximum(zero, torch.minimum(labels, preds)))
    FP = torch.sum(torch.maximum(zero, preds-labels))
    FN = torch.sum(torch.maximum(zero, labels-preds))
    return 2*TP / (2*TP + FP + FN)

# --------------------------------------------------------------------------------

def init_weights(module):
    with torch.no_grad():
        if (type(module) is nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.0)
        if (type(module) is nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            module.bias.data.fill_(0.0)
        if (type(module) is nn.BatchNorm1d):
            nn.init.uniform_(module.weight)
            module.bias.data.fill_(0.0)
        if (type(module) is nn.BatchNorm2d):
            nn.init.uniform_(module.weight)
            module.bias.data.fill_(0.0)

# --------------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, loss_fn):
    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        del loss
        del preds
    return running_loss / len(train_loader)
    
def valid_epoch(model, test_loader, loss_fn, output_transform):
    model.eval()
    running_loss = 0.0
    running_f1_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            running_loss += loss.item()
            running_f1_loss += f1_score(labels, preds, output_transform).item()
            del loss
            del preds
        f1 = running_f1_loss / len(test_loader)
        loss = running_loss / len(test_loader)
    return f1, loss

def train(model, train_loader, val_loader, nb_epochs, optimizer, scheduler, loss_fn, output_transform, checkpoint_path, show_plot=False):
    # Initialize variable to return
    best_model = model.state_dict()
    best_epoch = 0
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    val_f1s = []
    for epoch in tqdm(range(nb_epochs)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_f1, val_loss = valid_epoch(model, val_loader, loss_fn, output_transform)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        if (val_f1 > best_f1):
            best_model = model.state_dict()
            best_epoch = epoch + 1
            best_f1 = val_f1
            torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'scheduler': scheduler.state_dict()
            }, checkpoint_path)
        if show_plot:
            clear_output()
            plot_training(best_epoch, val_f1s, val_losses, train_losses, first_epoch_plot=1)
            print(f"Current train loss = {train_losses[-1]}, current validation loss = {val_losses[-1]}, current validation f1 loss = {val_f1s[-1]}")
    return best_model, best_f1, best_epoch, val_f1s, val_losses, train_losses

# --------------------------------------------------------------------------------

def plot_training(best_epoch, val_f1s, val_losses, train_losses, first_epoch_plot=1):
    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(val_f1s)+1)
    epoch_ticks_spacing = max(1, len(val_f1s) // 10)
    # Modify for plot
    epochs_plot = epochs[first_epoch_plot-1:]
    val_f1s_plot = val_f1s[first_epoch_plot-1:]
    val_losses_plot = val_losses[first_epoch_plot-1:]
    train_losses_plot = train_losses[first_epoch_plot-1:]
    # Plot F1 score
    ax[0].plot(epochs_plot, val_f1s_plot, label="Val")
    ax[0].vlines(best_epoch, ymin=np.min(val_f1s_plot), ymax=np.max(val_f1s_plot), color='k', ls='--', label="Best epoch")
    ax[0].set_xlabel("Training steps")
    ax[0].set_ylabel("F1-score")
    ax[0].set_title("F1-score")
    ax[0].xaxis.set_major_locator(MultipleLocator(epoch_ticks_spacing))
    ax[0].grid()
    ax[0].legend()
    # Plot losses
    ax[1].plot(epochs_plot, val_losses_plot, label="Val")
    ax[1].plot(epochs_plot, train_losses_plot, label="Train")
    ax[1].vlines(best_epoch, ymin=min(np.min(train_losses_plot), np.min(val_losses_plot)), ymax=max(np.max(train_losses_plot), np.max(val_losses_plot)), 
                 color='k', ls='--', label="Best epoch")
    ax[1].set_xlabel("Training steps")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Losses")
    ax[1].xaxis.set_major_locator(MultipleLocator(epoch_ticks_spacing))
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    plt.show()