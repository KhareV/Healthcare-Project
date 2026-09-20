from evaluation.metrics import PredictionRecord, evaluate_icu_time, evaluate_recovery_horizon


TEST_BOOTSTRAP_SEED = 8675309


def regression_record(stay, time, error, eligible=True):
    return PredictionRecord(stay, time, 0.0, float(error), eligible)


def support_record(stay, time, label, probability, eligible=True):
    return PredictionRecord(stay, time, float(label), float(probability), eligible)


def recovery_mae(records):
    return evaluate_recovery_horizon(records, horizon="24h").metrics["mae"]


def icu_median_ae(records):
    return evaluate_icu_time(records).metrics["median_absolute_error"]
