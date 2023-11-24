from bayes_conf_mat.metrics import IMPLEMENTED_SIMPLE_METRICS


def compute_all_simple_metrics(samples):
    simple_metrics = dict()
    for indentifier, simple_metric in filter(
        lambda x: x[1],
        list(IMPLEMENTED_SIMPLE_METRICS.items()),
    ):
        simple_metrics[indentifier] = simple_metric(samples)

    return simple_metrics
