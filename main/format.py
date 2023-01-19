POSTERIOR_COLUMN_ORDER = [
    "Instance",
    "N",
    "MAP",
    "Mean",
    "Std. Dev",
    "CI LB",
    "CI UB",
]

INDEP_TEST_COLUMN_ORDER = [
    "A Mean",
    "A CI LB",
    "A CI UB",
    "B Mean",
    "B CI LB",
    "B CI UB",
    "p(A>B)",
    "WSI LB",
    "WSI UB",
]


def find_format_backend(verbose: bool = False):

    try:
        import pandas as pd
    except ModuleNotFoundError:
        if verbose:
            print("Could not find pandas.")
    else:
        return "pandas"

    try:
        import tabulate
    except ModuleNotFoundError:
        if verbose:
            print("Could not find tabulate.")
    else:
        return "tabulate"

    if verbose:
        print("Falling back to json")
    return "json"


def flatten_summary(summary):

    records = []
    for k, v in summary.items():
        for kk, vv in summary[k].items():
            for k3, v3 in vv.items():
                records.append(
                    {
                        "Class": k,
                        "Metric": kk,
                        "Type": k3,
                        "Value": v3,
                    }
                )

    return records


def pandas_summary(records):

    import pandas as pd

    df = pd.DataFrame.from_records(records)

    df["Class"] = pd.Categorical(
        df["Class"], categories=df["Class"].unique(), ordered=True
    )
    df["Metric"] = pd.Categorical(
        df["Metric"], categories=df["Metric"].unique(), ordered=True
    )

    df = df.pivot(
        index=["Class", "Metric"],
        columns="Type",
        values="Value",
    )

    return df


def pandas_posterior_summary(records):

    df = pandas_summary(records)

    df = df[POSTERIOR_COLUMN_ORDER]

    return df


def pandas_indep_test_summary(records):

    df = pandas_summary(records)

    df = df[INDEP_TEST_COLUMN_ORDER]

    return df
