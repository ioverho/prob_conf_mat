import typing


def fmt(number: float, precision: int = 4):
    if round(number, precision) == 0:
        return f"{number:.2e}"
    else:
        return f"{number:{precision+2}.{precision}f}"


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
