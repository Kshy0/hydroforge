"""Device predicates shared by kernel binding and execution."""


def devices_match(actual, expected) -> bool:
    import torch

    actual = torch.device(actual)
    expected = torch.device(expected)
    return (
        actual.type == expected.type
        and (expected.index is None or actual.index == expected.index)
    )
