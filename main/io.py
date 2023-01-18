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
