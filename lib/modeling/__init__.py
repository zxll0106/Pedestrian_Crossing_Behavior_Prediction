from .rnn_based.model import ActionIntentionDetection as RNNModel


def make_model(cfg):

    model = RNNModel(cfg)


    return model
