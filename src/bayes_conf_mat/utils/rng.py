import collections.abc

import numpy as np


def seed_to_rng(
    seed: int
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | None,
) -> tuple[int, np.random.BitGenerator]:
    # Try to construct the RNG
    try:
        rng = np.random.default_rng(seed)
    except Exception as e:
        raise TypeError(
            f"Could not parse RNG from seed type {type(seed)}. Led to the following numpy error:\n{e}"
        )

    # Recover the initial seed
    parsed_seed = rng.bit_generator.seed_seq.entropy  # type: ignore

    if isinstance(parsed_seed, collections.abc.Iterable):
        raise TypeError("A sequence of ints as seed is currently not supported.")

    return parsed_seed, rng  # type: ignore
