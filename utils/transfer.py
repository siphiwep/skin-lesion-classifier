def set_non_trainable(model):
    for layer in model.layers:
        layer.trainable = False
    return model
