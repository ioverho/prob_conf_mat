import typing


def fmt(number: float, precision: int = 4, mode: str = "f") -> str:
    # Format as float, falling back to scientific notation if too small
    if mode == "f":
        if number != 0.0 and abs(number) <= 1 * (10 ** (-precision)):
            return f"{number:.{precision}e}"
        else:
            return f"{number:.{precision}f}"

    elif mode == "e":
        return f"{number:.{precision}e}"

    elif mode == "%":
        return f"{number * 100:2.{precision - 2}f}%"

    else:
        raise ValueError(
            f"Parameter mode must be one of 'f', 'e', '%'. Currently: {mode}"
        )


def summary_to_table_row(
    name: str, summary: typing.List[typing.Tuple], precision: int = 4
):
    row = [name]
    row += [
        f"[{fmt(val[0], precision=precision)}, {fmt(val[1], precision=precision)}]"
        if "HDI" in stat
        else val
        for stat, val in summary
    ]

    return row
