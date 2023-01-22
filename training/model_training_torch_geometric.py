import os
from datetime import datetime

import pytz
import torch
import torch.nn as nn
from ray.air import session
from torch import optim
from tqdm import tqdm

from config import device, CHECKPOINT_DIR
from graph.data_handling.dataset_featurizer import get_train_val_dataloaders, get_full_train_dataloader


def train(epoch, model, loss_function, optimizer, data_loader, hparams, search=False):
    model.train()
    model.to(device)
    train_loss = 0
    correct = 0
    count = 0

    # Iterate over the batches
    with tqdm(data_loader, unit='batch', disable=search) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for i_batch, batch in enumerate(tepoch):
            optimizer.zero_grad()
            output = model(batch.to(device))
            loss = loss_function(output, batch.y)
            train_loss += loss.detach().cpu().item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(batch.y)
            correct += torch.sum(preds.eq(batch.y).double()).detach().cpu().item()
            loss.backward()
            optimizer.step()
            loss_train = train_loss / count
            acc_train = correct / count
            tepoch.set_postfix(loss_train=loss_train, acc_train=acc_train)

    loss_train = train_loss / count
    return loss_train, acc_train



def validate(epoch, model, loss_function, data_loader, hparams, search=False):
    val_loss = 0
    correct = 0
    count = 0

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch.to(device))
            loss = loss_function(output, batch.y)
            val_loss += loss.detach().cpu().item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(batch.y)
            correct += torch.sum(preds.eq(batch.y).double()).detach().cpu().item()

        loss_val = val_loss / count
        acc_val = correct / count
        if not search:
            print('Epoch: {:03d}'.format(epoch),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                 )
    return loss_val, acc_val



def launch_experiment(model, hparams, experiment_name=None, seed=None, search=False, use_bert_embedding=False):
    if not search:
        if experiment_name is None:
            experiment_name = "experiment"
        experiment_name += f"_{datetime.now(tz=pytz.timezone('Europe/Paris')).strftime('%Y-%m-%d_%Hh%Mm%Ss')}"
        print(f"Launching new experiment {experiment_name}")
        print(f"Using {device}")

    train_data_loader, val_data_loader = get_train_val_dataloaders(hparams['batch_size'], val_size=0.20, random_state=seed, use_bert_embedding=use_bert_embedding)

    model.to(device)
    #optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=hparams['learning_rate'], momentum=0.9, nesterov=True, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    for epoch in range(hparams['epochs']):
        train_loss, train_acc = train(epoch, model, loss_function, optimizer, train_data_loader, hparams, search)
        val_loss, val_acc = validate(epoch, model, loss_function, val_data_loader, hparams, search)

        # Log to rayTune if doing hyperparameter search
        if search:
            session.report({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

        # Save checkpoint if val loss improved and if not in hyperaparameter search
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = f"{CHECKPOINT_DIR}/{experiment_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/model_epoch={epoch}_val-loss={val_loss:.3f}.pth"
            print(f"Validation loss improved, saving checkpoint to {save_path}")
            print()
            checkpoint = {
                'experiment_name': experiment_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss/train': train_loss,
                'loss/val': val_loss,
                'hparams': hparams
            }
            torch.save(checkpoint, save_path)
    return best_val_acc


def train_without_validation(model, hparams, epochs, experiment_name=None, seed=42):
    if experiment_name is None:
        experiment_name = "full_training"
        experiment_name += f"_{datetime.now(tz=pytz.timezone('Europe/Paris')).strftime('%Y-%m-%d_%Hh%Mm%Ss')}"
    print(f"Launching full training for {epochs} epochs: {experiment_name}")

    train_data_loader = get_full_train_dataloader(hparams['batch_size'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    loss_function = nn.CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train(epoch, model, loss_function, optimizer, train_data_loader, hparams, search=False)

    # Saving checkpoint
    save_dir = f"{CHECKPOINT_DIR}/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/FULL-model_epoch={epoch}_train-loss={train_loss:.3f}.pth"
    print(f"Training finished. Saving checkpoint to {save_path}")
    print()
    checkpoint = {
        'experiment_name': experiment_name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss/train': train_loss,
        'loss/val': None,
        'hparams': hparams
    }
    torch.save(checkpoint, save_path)
    return model

if __name__ == '__main__':
    # Example
    from graph.models.rgcn import RGCN, hparams
    from utils.submission import load_model

    model = RGCN(hparams)
    experiment_name = 'test'
    # model, hparams, experiment_name = load_model(RGCN, '/Users/amrictrudel/Documents/Repos/challenge/checkpoints/test_2023-01-20_23h47m33s/model_epoch=49_val-loss=1.942.pth')
    best_val_loss = launch_experiment(model, hparams, experiment_name=experiment_name, seed=42)
    # train_without_validation(model, hparams, 1, 'test_2023-01-20_22h43m43s')

    # generate_predictions(model, experiment_name)