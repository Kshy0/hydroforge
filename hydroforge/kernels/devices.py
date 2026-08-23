"""Device predicates shared by kernel binding and execution."""
import torch



def devices_match(actual: torch.device, expected: torch.device) -> bool:
    return (
        actual.type == expected.type
        and (expected.index is None or actual.index == expected.index)
    )
