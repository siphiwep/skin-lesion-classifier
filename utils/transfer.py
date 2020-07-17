def set_non_trainable(model):
    #[:18] Make first 18 layers trainable false 
    #[18:] make first 18 layers trainable true
    for layer in model.layers[:10]:
        layer.trainable = False
    return model
