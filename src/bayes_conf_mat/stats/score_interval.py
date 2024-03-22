from math import sqrt


def wilson_score_interval(p, n, z=1.96):
    # https://stackoverflow.com/a/74035575
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator

    return upper_bound - centre_adjusted_probability
