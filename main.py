from src.dataset import get_dataloaders
from src.model import CNN_TUMOR
from src.train import Train_Val
from src.evaluate import evaluate_model

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    model = CNN_TUMOR(params).to(device)
    model, loss_hist, metric_hist = Train_Val(model, params_train)
    evaluate_model(model, val_loader)
