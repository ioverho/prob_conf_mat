# Check if we can remove this function
def fmt(number: float, precision: int = 4, mode: str = "f") -> str:
    """Formats a float to use a certain precision."""
    # Format as float, falling back to scientific notation if too small
    if mode == "f":
        if number != 0.0 and abs(number) <= 1 * (10 ** (-precision)):
            return f"{number:.{precision}e}"
        return f"{number:.{precision}f}"

    if mode == "e":
        return f"{number:.{precision}e}"

    if mode == "%":
        return f"{number * 100:2.{precision - 2}f}%"

    raise ValueError(
        f"Parameter mode must be one of 'f', 'e', '%'. Currently: {mode}",
    )
