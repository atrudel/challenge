import csv

from config import OUTPUT_DIR


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
