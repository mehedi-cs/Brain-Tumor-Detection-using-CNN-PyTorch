import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

def true_and_pred(val_loader, model, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

def evaluate_model(val_loader, model, device, class_labels):
    y_true, y_pred = true_and_pred(val_loader, model, device)
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    show_confusion_matrix(cm, class_labels)
