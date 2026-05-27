"""
Federated cache merge for cross-device TranslationMemory sync.

Scope (be honest): this is *set merging of JSON snapshots*, not federated
learning. No gradients, no DP, no secure aggregation. The point is plumbing:
in a factory deployment where 50+ workers each carry a translator device,
each device's local TM accumulates Vi phrases that worker happens to use.
A nightly batch sync pools the snapshots so worker A picks up phrases
worker B has already paid the MT cost for. This belongs in the paper's
deployment-discussion section (§5), not the mechanism section.

Inputs: JSON files produced by `persistent_cache.save_t2_text_cache`.
Output: one unified TranslationMemory + a stats report.

Conflict policy:
  Two entries collide when their *normalized* Vi text + target_lang match
  exactly (same fingerprint key). Keep the entry with more `hits`
  (tie-break: earlier `inserted` timestamp — older entry has had more
  time to be validated). Different target_lang for same vi_text → both
  kept, they translate to different languages.

Anonymization:
  Current snapshot schema has no PII fields. `--anonymize` strips a
  documented forward-compat allow-list (`speaker_id`, `device_id`,
  `worker_id`, `site_id`) from each loaded record before merge so that
  if future snapshots embed those, this utility silently scrubs them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running this module as `python prototype/federated_cache.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translation_memory import TranslationMemory, _normalize  # noqa: E402
from persistent_cache import save_t2_text_cache               # noqa: E402


_ANON_FIELDS = ("speaker_id", "device_id", "worker_id", "site_id")


def _anonymize_record(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k not in _ANON_FIELDS}
    return out


def _load_snapshot_raw(path: Path) -> list[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("kind") != "t2_text":
        raise ValueError(f"{path}: not a t2_text snapshot (kind={obj.get('kind')!r})")
    return list(obj.get("entries", []))


def _entry_key(rec: dict) -> tuple[str, str]:
    return (_normalize(rec.get("vi_text", "")), rec.get("target_lang", ""))


def merge_tm_snapshots(
    snapshot_paths: list[Path],
    anonymize: bool = False,
    max_entries: int = 1024,
    cosine_hit: float = 0.80,
) -> tuple[TranslationMemory, dict]:
    """Merge multiple t2_text snapshot files into a single TranslationMemory.

    Returns (tm, stats_dict). stats_dict contains per-snapshot counts,
    overlap count, kept count, and merged total.
    """
    per_snapshot: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    overlap_keys: set[tuple[str, str]] = set()

    for p in snapshot_paths:
        recs = _load_snapshot_raw(Path(p))
        if anonymize:
            recs = [_anonymize_record(r) for r in recs]
        per_snapshot.append({
            "path": str(p),
            "n_entries": len(recs),
        })
        for rec in recs:
            key = _entry_key(rec)
            if not key[0]:
                continue
            if key in by_key:
                overlap_keys.add(key)
                kept = by_key[key]
                kept_hits = int(kept.get("hits", 0))
                new_hits  = int(rec.get("hits", 0))
                if new_hits > kept_hits:
                    by_key[key] = rec
                elif new_hits == kept_hits:
                    # tie-break: older insertion wins (more validated over time)
                    if float(rec.get("inserted", 0.0)) < float(kept.get("inserted", 0.0)):
                        by_key[key] = rec
            else:
                by_key[key] = rec

    # Build the merged TranslationMemory.
    tm = TranslationMemory(max_entries=max_entries, cosine_hit=cosine_hit)
    surviving = list(by_key.values())
    total_input_hits = 0
    for rec in surviving:
        tm.put(rec["vi_text"], rec["tgt_text"], rec["target_lang"], audio=None)
        # Patch hits on the most-recently-inserted store entry (LRU order).
        last_id = next(reversed(tm._store))
        tm._store[last_id]["hits"]     = int(rec.get("hits", 0))
        tm._store[last_id]["inserted"] = float(rec.get("inserted", time.time()))
        total_input_hits += int(rec.get("hits", 0))

    stats = {
        "n_snapshots":         len(snapshot_paths),
        "per_snapshot":        per_snapshot,
        "sum_input_entries":   sum(s["n_entries"] for s in per_snapshot),
        "n_overlapping_keys":  len(overlap_keys),
        "n_merged_entries":    len(surviving),
        "total_hits_preserved": total_input_hits,
        "anonymized":          bool(anonymize),
        "cosine_hit":          cosine_hit,
        "max_entries":         max_entries,
    }
    return tm, stats


# =====================================================================
# Demo: synthesize 3 mock worker snapshots, merge, report
# =====================================================================

def _make_mock_worker_snapshots(out_dir: Path) -> list[Path]:
    """Three workers, partially overlapping Vi phrase repertoires."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # shared "factory floor" phrases that any worker may have triggered
    shared = [
        ("an toàn lao động",        "occupational safety",        "en"),
        ("dừng dây chuyền ngay",    "stop the production line",   "en"),
        ("kiểm tra máy móc",        "inspect the machinery",      "en"),
    ]
    # worker-A: warehouse/logistics flavor
    worker_a = shared + [
        ("nâng hàng cẩn thận",      "lift the cargo carefully",   "en"),
        ("xe nâng đang chạy",       "forklift is moving",         "en"),
        ("kho số ba đầy rồi",       "warehouse three is full",    "en"),
        ("xin chào các bạn",        "hello everyone",             "en"),
    ]
    # worker-B: assembly line, shares 2 with A
    worker_b = shared[:2] + [
        ("hết vật liệu rồi",        "out of materials",           "en"),
        ("máy số hai bị lỗi",       "machine two has an error",   "en"),
        ("cần thay ca",             "shift change needed",        "en"),
        ("xin chào các bạn",        "hi everyone",                "en"),
    ]
    # worker-C: maintenance, shares 1 with A, 1 with B
    worker_c = [shared[0]] + [
        ("bảo trì định kỳ",         "scheduled maintenance",      "en"),
        ("thay dầu máy",            "change machine oil",         "en"),
        ("kiểm tra điện áp",        "check the voltage",          "en"),
        ("máy số hai bị lỗi",       "machine #2 fault",           "en"),
        ("hệ thống đã ổn định",     "system is stable",           "en"),
    ]

    # Build each worker's TM with realistic hit distributions.
    # Worker A uses 'an toàn lao động' a lot; worker B uses 'máy số hai bị lỗi' a lot.
    hit_profiles = [
        {"an toàn lao động": 12, "xe nâng đang chạy": 5, "xin chào các bạn": 3},   # A
        {"máy số hai bị lỗi": 9, "an toàn lao động": 4, "xin chào các bạn": 1},     # B
        {"bảo trì định kỳ": 7,  "máy số hai bị lỗi": 2, "an toàn lao động": 1},     # C
    ]

    out_paths = []
    for i, (corpus, hits_map) in enumerate(zip([worker_a, worker_b, worker_c], hit_profiles)):
        tm = TranslationMemory(max_entries=64, cosine_hit=0.80)
        for vi, tgt, lang in corpus:
            tm.put(vi, tgt, lang, audio=None)
        # patch hits
        for key, e in tm._store.items():
            h = hits_map.get(e["vi_text"], 0)
            e["hits"] = h

        p = out_dir / f"worker_{chr(ord('A')+i)}_t2.json"
        save_t2_text_cache(tm, p)
        out_paths.append(p)
    return out_paths


def _demo(results_path: Path) -> None:
    snap_dir = Path(__file__).resolve().parents[1] / "data" / "cache_snapshots" / "_federated_mock"
    snapshot_paths = _make_mock_worker_snapshots(snap_dir)

    tm, stats = merge_tm_snapshots(snapshot_paths, anonymize=True)

    # Hit-rate-on-merge: simulate every worker's vocabulary as queries against
    # the merged TM. A miss means "this phrase wasn't in any worker's snapshot",
    # which is impossible by construction here — so hit-rate = 100% is the
    # honest expected baseline. The interesting number is the *cross-worker*
    # hit rate: query worker A's phrases against the merged TM and count how
    # many would have come from a *different* worker.

    # Re-load each snapshot raw to know origin of each phrase.
    origin: dict[str, set[str]] = {}
    for p in snapshot_paths:
        worker_id = Path(p).stem
        for rec in _load_snapshot_raw(p):
            origin.setdefault(_normalize(rec["vi_text"]), set()).add(worker_id)

    cross_worker = {}
    for p in snapshot_paths:
        worker_id = Path(p).stem
        own_lookups = 0
        cross_hits  = 0
        for rec in _load_snapshot_raw(p):
            own_lookups += 1
            owners = origin.get(_normalize(rec["vi_text"]), set())
            if owners - {worker_id}:
                cross_hits += 1
        cross_worker[worker_id] = {
            "queries":            own_lookups,
            "cross_worker_hits":  cross_hits,
            "cross_worker_rate":  round(cross_hits / max(own_lookups, 1), 4),
        }

    report = {
        "demo":              "federated TM snapshot merge (3 mock workers)",
        "snapshot_paths":    [str(p) for p in snapshot_paths],
        "merge_stats":       stats,
        "merged_tm_summary": tm.summary(),
        "cross_worker_benefit": cross_worker,
        "honest_scope_note":
            "This utility performs set-union merging of JSON cache snapshots, "
            "not federated learning. No model parameters are exchanged. "
            "Cited as deployment infrastructure (§5), not a research mechanism.",
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console pretty-print
    print("=" * 64)
    print("Federated TM Snapshot Merge — Demo")
    print("=" * 64)
    print(f"Snapshots merged       : {stats['n_snapshots']}")
    for s in stats["per_snapshot"]:
        print(f"  {Path(s['path']).name:30s}  {s['n_entries']:>3} entries")
    print(f"Sum of input entries   : {stats['sum_input_entries']}")
    print(f"Overlapping vi keys    : {stats['n_overlapping_keys']}")
    print(f"Merged total entries   : {stats['n_merged_entries']}")
    print(f"Compression vs sum     : {stats['n_merged_entries']}/{stats['sum_input_entries']} "
          f"= {stats['n_merged_entries']/max(stats['sum_input_entries'],1):.0%}")
    print(f"Hits preserved (total) : {stats['total_hits_preserved']}")
    print(f"Anonymized             : {stats['anonymized']}")
    print()
    print("Cross-worker benefit (phrases now available from peers):")
    for w, c in cross_worker.items():
        print(f"  {w:25s} {c['cross_worker_hits']:>2}/{c['queries']:<2}  "
              f"= {c['cross_worker_rate']:.0%} of own vocab also held by a peer")
    print()
    print(f"Report written to: {results_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Merge t2_text TranslationMemory JSON snapshots.")
    ap.add_argument("snapshots", nargs="*", type=Path,
                    help="paths to t2_text JSON snapshots (omit to run demo)")
    ap.add_argument("--anonymize", action="store_true",
                    help="strip forward-compat PII fields before merge")
    ap.add_argument("--out", type=Path, default=None,
                    help="write merged TM JSON to this path (uses save_t2_text_cache)")
    ap.add_argument("--demo", action="store_true",
                    help="run the 3-mock-worker demo and write Results/federated_cache_demo.json")
    args = ap.parse_args()

    if args.demo or not args.snapshots:
        out = Path(__file__).resolve().parents[1] / "Results" / "federated_cache_demo.json"
        _demo(out)
        return

    tm, stats = merge_tm_snapshots(args.snapshots, anonymize=args.anonymize)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.out:
        info = save_t2_text_cache(tm, args.out)
        print(f"\nMerged TM saved: {info}")


if __name__ == "__main__":
    main()
