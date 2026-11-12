"""Recorded training protocol and strict completed-run reuse checks.

This module uses only the Python standard library. It preserves the existing
config.json schema; the guard compares recorded fields, not arbitrary source
code or library-build changes that were never recorded in historical configs.
"""
from collections.abc import Mapping, Sequence


BATCH_SIZE = 64
INPUT_SIZE = 128
WEIGHT_DECAY = 0.0001
CHECKPOINT_RULE = 'maximum validation macro_f1; earliest on tie'
METHODS = ('cnn', 'resnet18', 'resnet18_aug')
PROTOCOL_FIELDS = (
    'method', 'seed', 'epochs', 'batch_size', 'input_size', 'lr',
    'weight_decay', 'optimizer', 'scheduler', 'pretrained', 'classes',
    'parameters', 'checkpoint_rule', 'manifest_sha256',
)


class ProtocolMismatchError(ValueError):
    """A completed run cannot safely be reused for the requested protocol."""


def learning_rate_for_method(method: str) -> float:
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method!r}; expected one of {METHODS!r}')
    return 0.001 if method == 'cnn' else 0.0003


def build_training_protocol(
    *,
    method: str,
    seed: int,
    epochs: int,
    classes: Sequence[str],
    parameters: int,
    manifest_sha256: str,
) -> dict[str, object]:
    """Return the existing configuration fields and unchanged numeric settings.

    ``parameters`` must come from the actual current model, not the saved run.
    ``classes`` retains its order because this order defines classifier IDs.
    """
    return {
        'method': method,
        'seed': seed,
        'epochs': epochs,
        'batch_size': BATCH_SIZE,
        'input_size': INPUT_SIZE,
        'lr': learning_rate_for_method(method),
        'weight_decay': WEIGHT_DECAY,
        'optimizer': 'AdamW',
        'scheduler': 'cosine',
        'pretrained': method != 'cnn',
        'classes': list(classes),
        'parameters': parameters,
        'checkpoint_rule': CHECKPOINT_RULE,
        'manifest_sha256': manifest_sha256,
    }


def validate_completed_run_config(
    actual: object,
    expected: Mapping[str, object],
    *,
    run_name: str = '',
) -> None:
    """Raise on missing, changed or differently typed protocol values.

    Extra non-protocol metadata may remain in a historical configuration.
    Explicit exceptions keep these checks active under ``python -O``.
    """
    if not isinstance(actual, Mapping):
        raise ProtocolMismatchError('Completed run configuration must be a mapping')
    mismatches = []
    for field in PROTOCOL_FIELDS:
        if field not in actual:
            mismatches.append(f'{field}: missing')
            continue
        required = expected[field]
        recorded = actual[field]
        if type(recorded) is not type(required) or recorded != required:
            mismatches.append(f'{field}: recorded={recorded!r}, expected={required!r}')
    if mismatches:
        name = f' {run_name!r}' if run_name else ''
        raise ProtocolMismatchError(
            f'Refusing to reuse completed run{name}; protocol mismatch: '
            + '; '.join(mismatches)
            + '. Keep the historical outputs and use a new run directory.'
        )
