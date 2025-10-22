def Train_Val(model, params, device, verbose=False):
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        current_lr = get_lr(opt)
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        metric_history["train"].append(train_metric)
        metric_history["val"].append(val_metric)

        lr_scheduler.step(val_loss)
        model.load_state_dict(best_model_wts)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | LR={current_lr:.6f} | Train Loss={train_loss:.4f} | "
                  f"Val Loss={val_loss:.4f} | Val Acc={100*val_metric:.2f}%")

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
