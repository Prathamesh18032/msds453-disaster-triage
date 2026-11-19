"""Fixed MinHash candidate retrieval, exact confirmation, and component reduction."""
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version

from .data_policy import row_order
from .provenance import sha256_bytes


@lru_cache(maxsize=1024)
def shingles(text):
    return frozenset(text[i:i + 5] for i in range(len(text) - 4))


def qualifies(left, right):
    union = len(left | right)
    return union > 0 and 10 * len(left & right) >= 9 * union


def duplicate_components(rows):
    from datasketch import MinHash, MinHashLSH
    if version("datasketch") != "2.0.0":
        raise ValueError("duplicate preparation requires datasketch 2.0.0")
    rows = sorted(rows, key=row_order)
    by_key = {r.source_key: r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate source occurrence")
    parent = {key: key for key in by_key}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    edges = []

    def edge(a, b, reason):
        left, right = sorted((a, b))
        edges.append({"left": left, "right": right, "reason": reason})
        x, y = find(left), find(right)
        if x != y:
            parent[max(x, y)] = min(x, y)

    ids, texts = {}, {}
    for row in rows:
        for mapping, value, reason in ((ids, row.numeric_id, "identical_id"),
                                       (texts, row.classical_text, "identical_normalized_text")):
            if value in mapping:
                edge(mapping[value], row.source_key, reason)
            else:
                mapping[value] = row.source_key
    lsh = MinHashLSH(threshold=.8, num_perm=128, weights=(.5, .5))
    candidate_count = confirmed_count = 0
    permutation_sha256 = None
    for row in rows:
        if len(row.classical_text) < 30:
            continue
        grams = shingles(row.classical_text)
        signature = MinHash(num_perm=128, seed=42)
        signature.update_batch([g.encode("utf-8") for g in sorted(grams)])
        if permutation_sha256 is None:
            permutation_sha256 = sha256_bytes(signature.permutations.astype("<u8").tobytes())
        for other_key in sorted(lsh.query(signature)):
            candidate_count += 1
            if qualifies(grams, shingles(by_key[other_key].classical_text)):
                edge(other_key, row.source_key, "confirmed_near_duplicate")
                confirmed_count += 1
        lsh.insert(row.source_key, signature)
    grouped = defaultdict(list)
    for row in rows:
        grouped[find(row.source_key)].append(row)
    kept, excluded, groups = [], [], []
    for values in grouped.values():
        values = sorted(values, key=lambda r: (r.numeric_id, r.member, r.line))
        representative = values[0]
        group_id = f"humaid-group:{representative.numeric_id}"
        reasons = []
        if len({r.event_id for r in values}) > 1:
            reasons.append("cross_event_component")
        if len({r.y_true for r in values}) > 1:
            reasons.append("conflicting_label_component")
        groups.append({"group_id": group_id, "minimum_numeric_id": str(representative.numeric_id),
                       "members": sorted(r.source_key for r in values),
                       "representative": None if reasons else representative.source_key,
                       "exclusion_reasons": reasons})
        for row in values:
            if reasons or row != representative:
                excluded.append({**row.metadata(), "stage": "duplicates", "group_id": group_id,
                                 "reasons": reasons or ["duplicate_nonrepresentative"]})
            else:
                kept.append((row, group_id))
    metadata = {"implementation": "datasketch", "version": version("datasketch"),
                "hashfunc": "datasketch.hashfunc.sha1_hash32", "seed": 42, "num_perm": 128,
                "permutation_sha256": permutation_sha256, "lsh_threshold": .8,
                "lsh_weights": [.5, .5], "bands": lsh.b, "rows_per_band": lsh.r,
                "minimum_length": 30, "shingle_length": 5, "exact_jaccard_threshold": .9,
                "candidate_pairs": candidate_count, "confirmed_near_pairs": confirmed_count,
                "total_edges": len(edges), "components": len(groups),
                "limitation": "approximate candidate retrieval can miss qualifying pairs"}
    shingles.cache_clear()
    return (sorted(kept, key=lambda item: row_order(item[0])),
            sorted(excluded, key=lambda x: x["source_key"]),
            sorted(groups, key=lambda x: int(x["minimum_numeric_id"])),
            sorted(edges, key=lambda x: (x["left"], x["right"], x["reason"])), metadata)
