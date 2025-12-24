from sklearn.metrics import accuracy_score


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    model.fit(X_train, y_train)

    metrics = {}
    if X_val is not None and y_val is not None:
        preds = model.predict(X_val)
        metrics["accuracy"] = accuracy_score(y_val, preds)

    return model, metrics
