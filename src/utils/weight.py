from collections import OrderedDict


def averaging_weight(weight_list: list, model_name: str) -> dict:

    sum_weights = OrderedDict()
    is_updated = False

    for w in weight_list:

        weight = w[model_name]

        if not is_updated:
            sum_weights.update(weight)
            is_updated = True
            continue

        for weight_key in weight.keys():
            sum_weights[weight_key] += weight[weight_key]

    for weight_key in sum_weights.keys():
        sum_weights[weight_key] /= len(weight_list)

    return sum_weights
