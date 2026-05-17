from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn

def f1_score(labels, preds, output_transform):
    preds = output_transform(preds)
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

def train_epoch(model, train_dataset, optimizer, loss_fn, batch_size, batch_size_simul=1, epoch_permutation=None):
    if (epoch_permutation == None):
        epoch_permutation = torch.arange(len(train_dataset))
    running_loss = 0.0
    model.train()
    optimizer.zero_grad()
    batched_inputs = None
    batched_labels = None
    for i, sample_idx in enumerate(epoch_permutation.tolist()):
        if (i % batch_size_simul == 0):
            batched_inputs, batched_labels = train_dataset.__getitem__(sample_idx)
            batched_inputs = batched_inputs.unsqueeze(0)
            batched_labels = batched_labels.unsqueeze(0)
        else:
            next_inputs, next_labels = train_dataset.__getitem__(sample_idx)
            batched_inputs = torch.cat((batched_inputs, next_inputs.unsqueeze(0)), dim=0)
            batched_labels = torch.cat((batched_labels, next_labels.unsqueeze(0)), dim=0)
        if ((i + 1) % batch_size_simul == 0):
            preds = model(batched_inputs)
            loss = loss_fn(preds, batched_labels)
            loss.backward()
            running_loss += loss.item()
            del loss
            del preds
        if ((i + 1) % batch_size == 0):
            optimizer.step()
            optimizer.zero_grad()
    return running_loss * batch_size_simul / len(train_dataset)
    
    
def valid_epoch(model, val_dataset, loss_fn, output_transform):
    model.eval()
    running_loss = 0.0
    running_f1_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataset:
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            running_loss += loss.item()
            running_f1_loss += f1_score(labels, preds, output_transform).item()
            del loss
            del preds
        f1 = running_f1_loss / len(val_dataset)
        loss = running_loss / len(val_dataset)
    return f1, loss

def train(model, train_dataset, val_dataset, batch_size, nb_epochs, optimizer, scheduler, loss_fn, 
          output_transform, checkpoint_path, show_plot=False, batch_size_simul=1, generator=torch.Generator().manual_seed(42)):
    # Initialize variable to return
    best_model = model.state_dict()
    best_epoch = 0
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    val_f1s = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_permutation = torch.randperm(len(train_dataset), generator=generator)
        train_loss = train_epoch(model, train_dataset, optimizer, loss_fn, batch_size, 
                                 batch_size_simul=batch_size_simul, epoch_permutation=epoch_permutation)
        val_f1, val_loss = valid_epoch(model, val_dataset, loss_fn, output_transform)
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

def plot_training(best_epoch, val_f1s, val_losses, train_losses, first_epoch_plot=1, save_filename=None):
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
    ax[1].vlines(best_epoch, 
                 ymin=min(np.min(train_losses_plot), np.min(val_losses_plot)), 
                 ymax=max(np.max(train_losses_plot), np.max(val_losses_plot)), 
                 color='k', ls='--', label="Best epoch")
    ax[1].set_xlabel("Training steps")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Losses")
    ax[1].xaxis.set_major_locator(MultipleLocator(epoch_ticks_spacing))
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    if (save_filename != None):
        plt.savefig(save_filename)
    plt.show()