import csv

from config import OUTPUT_DIR
from graph.data_handling.dataset_featurizer import get_test_dataloader
from datetime import datetime
import pytz
import torch
from config import device


def write_submission_file(y_pred_proba, proteins_test, filename='submission.csv'):
    assert y_pred_proba.shape == (1223, 18), f"Incorrect dimensions for predict_proba: {y_pred_proba.shape}. Expected (1224, 18)."
    filepath = f'{OUTPUT_DIR}/{filename}'
    print(f"Generating submission file: {filepath}")
    with open(filepath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = list()
        for i in range(18):
            lst.append('class'+str(i))
        lst.insert(0, "name")
        writer.writerow(lst)
        for i, protein in enumerate(proteins_test):
            lst = y_pred_proba[i,:].tolist()
            lst.insert(0, protein)
            writer.writerow(lst)


def generate_predictions(model, experiment_name, batch_size=32, use_bert_embedding=False):
    test_dataloader = get_test_dataloader(batch_size, use_bert_embedding=use_bert_embedding)
    model.eval()
    model.to(device)
    y_pred_proba = list()
    protein_names = list()

    with torch.no_grad():
        for batch in test_dataloader:
            output = model(batch.to(device))
            y_pred_proba.append(output)
            protein_names += batch.protein_names
    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_pred_proba = torch.exp(y_pred_proba)
    y_pred_proba = y_pred_proba.detach().cpu().numpy()
    filename = f"{experiment_name}_submission.csv"
    write_submission_file(y_pred_proba, protein_names, filename)

def load_model(model_class, model_path):
    print(f"Loading pretrained model from {model_path}_submission.csv")
    checkpoint = torch.load(model_path)
    model = model_class(checkpoint['hparams']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    hparams = checkpoint['hparams']
    experiment_name = checkpoint['experiment_name']
    return model, hparams, experiment_name

if __name__ == '__main__':
    from graph.models.gat import GAT
    model_path = "checkpoints/test_gat_2023-01-21_19h31m08s/model_epoch=0_val-loss=3.453.pth"
    model, hparams, experiment_name = load_model(GAT, model_path)
    generate_predictions(model, experiment_name)