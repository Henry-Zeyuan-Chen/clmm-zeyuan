"""Deterministic two-fact data for an association overfit diagnostic.

The fixed bank balances both key exposure and the 16 answer classes. Fresh
assignments distinguish context recall from a persistent key-to-value lookup.
All examples use canonical template zero, two facts, and no replay or revision.
"""

from collections import defaultdict

import numpy as np

from .fact_streams import Episode, FactQuery, FactRecord, KEY_POOLS, VALUES


_MAX_ATTEMPTS = 10_000
_BANK_DOMAIN = 1
_SAME_DOMAIN = 2
_HELDOUT_DOMAIN = 3
_TRAINING_DOMAIN = 4


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _pair(left, right):
    """An injective pairing of nonnegative integers, without hash collisions."""
    total = left + right
    return total * (total + 1) // 2 + right


def _episode_seed(seed, domain, index, attempt=0):
    # The domain, accepted bank attempt (or training step), and row index are
    # explicit parts of the identity. Surface family never affects this seed.
    return _pair(_pair(_pair(seed, domain), attempt), index)


def _episode(seed, keys, values, split="train", family="seen"):
    events = tuple(FactRecord(index, key, value, 1, "fact", family, 0)
                   for index, (key, value) in enumerate(zip(keys, values)))
    queries = tuple(FactQuery(record.key, record.value, None, "stable",
                              record.event_id, record.event_id, family, 0)
                    for record in events)
    return Episode(seed, 2, 0, split, family, 0.0, False, events, queries)


def _family(family):
    if family not in ("seen", "heldout"):
        raise ValueError("family must be seen or heldout")


def _bank_episodes(bank):
    if not isinstance(bank, dict) or not isinstance(bank.get("episodes"), list):
        raise ValueError("bank must contain an episodes list")
    if not bank["episodes"]:
        raise ValueError("bank must contain at least one episode")
    episodes = [Episode.from_dict(data) for data in bank["episodes"]]
    if any((episode.split, episode.template_family, episode.facts, episode.delay,
            episode.revision_fraction, episode.include_recent) !=
           ("train", "seen", 2, 0, 0.0, False) for episode in episodes):
        raise ValueError("bank episodes must be seen two-fact training contexts")
    return episodes


def _bank_pool(bank):
    _bank_episodes(bank)
    pool = bank.get("key_pool")
    if (not isinstance(pool, list) or len(pool) < 2 or len(set(pool)) != len(pool)
            or not set(pool) <= set(KEY_POOLS["train"])):
        raise ValueError("bank key_pool must contain distinct training keys")
    return pool


def build_bank(seed=90201, contexts=32, key_count=16):
    """Return a JSON-ready, balanced fixed training bank.

    Each pass permutes the first ``key_count`` fixed training keys and pairs
    adjacent entries. A complete key/value plan is rejected if ordered pairs
    repeat, a context repeats a value, or a key has only one assigned value.
    ``attempt`` is the zero-based accepted rejection-sampling attempt.
    """
    _integer("seed", seed)
    _integer("contexts", contexts, 8)
    _integer("key_count", key_count, 4)
    if key_count % 2 or key_count > len(KEY_POOLS["train"]):
        raise ValueError("key_count must be even and within the training key pool")
    records = contexts * 2
    if records % len(VALUES):
        raise ValueError("contexts * 2 must be divisible by the 16 answer classes")
    if records % key_count:
        raise ValueError("contexts * 2 must be divisible by key_count")
    passes = records // key_count
    if passes < 2:
        raise ValueError("each key needs at least two appearances for distinct assignments")
    if contexts > key_count * (key_count - 1):
        raise ValueError("contexts exceeds the number of unique ordered key pairs")

    pool = list(KEY_POOLS["train"][:key_count])
    rng = np.random.default_rng(np.random.SeedSequence([seed, _BANK_DOMAIN]))
    balanced_values = list(VALUES) * (records // len(VALUES))
    for attempt in range(_MAX_ATTEMPTS):
        keys = [key for _ in range(passes) for key in rng.permutation(pool).tolist()]
        values = rng.permutation(balanced_values).tolist()
        pairs = list(zip(keys[::2], keys[1::2]))
        assignments = defaultdict(set)
        for key, value in zip(keys, values):
            assignments[key].add(value)
        if (len(set(pairs)) != contexts
                or any(left == right for left, right in zip(values[::2], values[1::2]))
                or any(len(assigned) < 2 for assigned in assignments.values())):
            continue
        episodes = [_episode(_episode_seed(seed, _BANK_DOMAIN, index, attempt),
                             pair, values[index * 2:index * 2 + 2])
                    for index, pair in enumerate(pairs)]
        return {"episodes": [episode.to_dict() for episode in episodes],
                "key_pool": pool, "seed": seed, "contexts": contexts,
                "key_count": key_count, "attempt": attempt}
    raise RuntimeError(f"Could not build a balanced association bank in {_MAX_ATTEMPTS} attempts")


def fresh_same(bank, seed, count=128, family="seen"):
    """Cycle bank contexts with new values excluding both original values.

    Training-namespace episodes support seen templates only under the fixed
    fact-stream protocol. Use ``fresh_heldout`` for paired template families.
    """
    _integer("seed", seed)
    _integer("count", count, 1)
    _family(family)
    if family != "seen":
        raise ValueError("fresh_same supports seen templates only for training keys")
    originals = _bank_episodes(bank)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _SAME_DOMAIN]))
    episodes = []
    for index in range(count):
        original = originals[index % len(originals)]
        alternatives = [value for value in VALUES
                        if value not in {record.value for record in original.events}]
        values = rng.choice(alternatives, size=2, replace=False).tolist()
        episodes.append(_episode(_episode_seed(seed, _SAME_DOMAIN, index),
                                 [record.key for record in original.events], values))
    return episodes


def fresh_heldout(seed, count=128, family="seen"):
    """Return fresh dev-key associations paired across seen/heldout wording."""
    _integer("seed", seed)
    _integer("count", count, 1)
    _family(family)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _HELDOUT_DOMAIN]))
    episodes = []
    for index in range(count):
        keys = rng.choice(KEY_POOLS["dev"], size=2, replace=False).tolist()
        values = rng.choice(list(VALUES), size=2, replace=False).tolist()
        episodes.append(_episode(_episode_seed(seed, _HELDOUT_DOMAIN, index),
                                 keys, values, "dev", family))
    return episodes


def fresh_training(bank, seed, step, count=8):
    """Sample a training step independently from ``SeedSequence([seed, step])``."""
    _integer("seed", seed)
    _integer("step", step)
    _integer("count", count, 1)
    pool = _bank_pool(bank)
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    episodes = []
    for index in range(count):
        keys = rng.choice(pool, size=2, replace=False).tolist()
        values = rng.choice(list(VALUES), size=2, replace=False).tolist()
        episodes.append(_episode(_episode_seed(seed, _TRAINING_DOMAIN, index, step),
                                 keys, values))
    return episodes
