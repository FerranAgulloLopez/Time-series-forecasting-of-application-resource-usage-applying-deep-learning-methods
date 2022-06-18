from functools import partial


def preprocess_model(preprocessing_steps, model):
    for block in model._modules:
        for step in preprocessing_steps:
            name = step['name']
            if name == 'weight_init':
                model._modules[block].apply(partial(weights_init, config=step))
            else:
                raise Exception('The preprocessing step \'' + step + '\' is unknown')


def weights_init(model, config):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
        _type = config['type']
        args = config['args']
        if _type == 'gaussian':
            if hasattr(model, 'weight'):
                model.weight.data.normal_(**args)
            if hasattr(model, 'bias') and model.bias is not None:
                model.bias.data.normal_(**args)
        elif _type == 'xavier_normal':
            if hasattr(model, 'weight'):
                model.weight.data.xavier_normal_(**args)
            if hasattr(model, 'bias') and model.bias is not None:
                model.bias.data.xavier_normal_(**args)
        elif _type == 'kaiming_normal_':
            if hasattr(model, 'weight'):
                model.weight.data.kaiming_normal_(**args)
            if hasattr(model, 'bias') and model.bias is not None:
                model.bias.data.kaiming_normal_(**args)
        else:
            raise Exception('Not recognized weight init')
        if classname.find('BatchNorm') != -1:
            model.bias.data.fill_(0)
