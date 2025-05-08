from __future__ import annotations

import typing
import collections.abc

import numpy as np

class RNG:

    def __init__(self, seed: int | typing.Sequence[int] | None, position: tuple = ()) -> None:
        """A wrapper for Numpy's Generator class, that tracks any spawned child RNGs.

        This way, we can change the RNG in a reproducible manner, on the fly.

        Args:
            seed (int | typing.Sequence[int] | None): A seed to initialize the `BitGenerator`. If None, then fresh, unpredictable entropy will be pulled from the OS. If an `int` or `array_like[ints]` is passed, then all values must be non-negative and will be passed to `SeedSequence` to derive the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance. Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`. If passed a `Generator`, it will be returned unaltered. When passed a legacy `RandomState` instance it will be coerced to a `Generator`.
            position (tuple, optional): the position of the RNG in the tree. Typically this is not set by the user. Defaults to a root node.
        """

        # The current seed & position
        self._seed = seed
        self.position = position

        # The current seed sequence
        self.seed_sequence: np.random.SeedSequence = np.random.SeedSequence(entropy=self._seed, spawn_key=self.position)

        # The RNG object
        self.rng: np.random.Generator = np.random.default_rng(seed=self.seed_sequence)

        # Any RNGs spawned from this RNG
        self.children: list[typing.Self] = []

    def __getattr__(self, name):
        """Called for failed attribute accesses."""
        return self.rng.__getattribute__(name)

    def spawn(self, n_children: int) -> list[typing.Self]: # type: ignore
        """Spawns a new independent RNG. It has a position lower in the tree, but shares the seed of its parent.

        Args:
            n_children (int): the number of children to spawn

        Returns:
            list[typing.Self]: a list of the newly generated child RNGs
        """

        cur_num_children = len(self.children)

        spawned_rngs: list[typing.Self] = []
        for i in range(n_children):
            new_rng = RNG(seed=self.seed, position=(*self.position, cur_num_children+i))

            spawned_rngs.append(new_rng) # type: ignore

        self.children.extend(spawned_rngs)

        return spawned_rngs

    @property
    def seed(self) -> int | typing.Sequence[int] | None:
        return self._seed

    @seed.setter
    def seed(self, value: int | typing.Sequence[int] | None) -> None:
        """Changes the seed without changing the position.

        Args:
            value (int | Sequence[int] | None): the need seed value
        """

        self._seed = value

        self.seed_sequence = np.random.SeedSequence(entropy=self._seed, spawn_key=self.position)

        self.rng = np.random.default_rng(seed=self.seed_sequence)

        for child in self.children:
            child.seed = value

    def __repr__(self) -> str:
        return f"RNG(seed={self.seed}, position={self.position})"

    def __str__(self) -> str:
        return f"RNG(seed={self.seed}, position={self.position})"
