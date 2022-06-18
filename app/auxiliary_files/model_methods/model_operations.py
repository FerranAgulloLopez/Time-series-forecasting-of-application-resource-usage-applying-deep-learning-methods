def model_arq_to_json(model) -> dict:
    dic = {}
    for key, value in model._modules.items():
        if len(value._modules) == 0:
            dic[key] = str(value)
        else:
            dic[key] = model_arq_to_json(value)
    return dic
