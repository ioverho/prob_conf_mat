def fmt(number: float, precision: int = 4, mode: str = "f"):
    # Format as float, falling back to scientific notation if too small
    if mode == "f":
        if round(number, precision) == 0:
            return f"{number:.{precision}e}"
        else:
            return f"{number:.{precision}f}"

    elif mode == "s":
        return f"{number:.{precision}e}"

    elif mode == "%":
        return f"{number*100:2.{precision-2}f}%"
