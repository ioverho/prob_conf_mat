import pickle
from pathlib import Path

IMPLEMENTED_FILE_TYPES = {".pt", ".ckpt"}


def load_file(fp):

    file_type = fp.suffix

    if file_type not in IMPLEMENTED_FILE_TYPES:
        raise ValueError(
            f"Could not load {fp}. File type not recognized. File must be one of {IMPLEMENTED_FILE_TYPES}"
        )

    elif file_type in {".pt", ".ckpt"}:

        import torch

        output = torch.load(
            fp,
            map_location="cpu",
        ).numpy()

        return output


class ObjectIO(object):
    def __init__(self, **kwargs):

        self._hparams = dict(kwargs)

    @property
    def _save_str(self):
        raise NotImplementedError

    def save(self, save_dir):

        save_dir = Path(save_dir)

        if save_dir.is_dir():
            with open(save_dir / self._save_str, "wb") as f:
                pickle.dump(self.__dict__, f)
        else:
            raise NotADirectoryError(save_dir)

        if self.verbose:
            print(f"Saved to {save_dir / self._save_str}")

    @classmethod
    def load(cls, fp):

        fp = Path(fp)

        if fp.exists():
            with open(fp, "rb") as f:
                state_dict = pickle.load(f)
        else:
            raise ValueError(f"No file found at:\n\t{fp}")

        instance = cls(**state_dict["_hparams"])

        for k, v in state_dict.items():
            if k == "_hparams":
                continue

            instance.__setattr__(k, v)

        if instance.verbose:
            print(f"Loaded from {fp}")

        return instance
