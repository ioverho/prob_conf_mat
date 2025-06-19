from math import sqrt


def wilson_score_interval(p: float, n: int, z: float = 1.96) -> float:
    """Estimates the radius of the intveral around a proprotion using the Wilson score interval.

    Adapted from [Stack Overflow](https://stackoverflow.com/a/74035575)

    Args:
        p (float): the relative number of wins
        n (int): the number of total trials
        z (float, optional): the critical value. Defaults to 1.96.

    Returns:
        float: half of the size of the iterval
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator

    return upper_bound - centre_adjusted_probability
