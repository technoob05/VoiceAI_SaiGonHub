"""
Benchmark: batched NLLB MT (translate_sentences_batch) vs sequential per-sentence
translate(). Pure-CPU, beam=1. Verifies byte-identical output and reports p50.

5 utterances: 1 short single-sent, 2 medium two-sent, 2 long three-sent.
Each utterance is translated to English; we measure wall time spent inside MT
only, repeating each utterance N_REPEAT times to suppress jitter.

Writes Results/mt_batch_bench.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

from pipeline_fast import FastCascadeS2ST, FastPipelineConfig, _split_sentences


UTTERANCES = [
    ("short_1sent",
     "Tôi tên là Minh."),
    ("medium_2sent_a",
     "Xin chào. Tôi tên là Minh."),
    ("medium_2sent_b",
     "Nhưng tôi thích xem bộ phim này. Phim này phát ngay sau chương trình."),
    ("long_3sent_a",
     "Một bà sang trọng đến gặp bác sĩ trẻ. Bác sĩ khám bệnh cho bà ấy. Bác sĩ hỏi bà có khoẻ không."),
    ("long_3sent_b",
     "Cây cổ thụ ấy bất chợt bị bệnh tật quật ngã. Người dân trong làng rất buồn. Họ quyết định trồng cây mới."),
]

N_REPEAT = 5
TARGET = "en"


def time_sequential(pipe: FastCascadeS2ST, sentences: list[str]) -> tuple[float, str]:
    t0 = time.perf_counter()
    parts: list[str] = []
    for s in sentences:
        if not s.strip():
            continue
        parts.append(pipe.translate(s, TARGET, source="vi"))
    dt = (time.perf_counter() - t0) * 1000
    return dt, " ".join(p for p in parts if p).strip()


def time_batched(pipe: FastCascadeS2ST, sentences: list[str]) -> tuple[float, str]:
    t0 = time.perf_counter()
    outs = pipe.translate_sentences_batch(sentences, TARGET, source="vi")
    dt = (time.perf_counter() - t0) * 1000
    text = " ".join(o for o in outs if o).strip()
    return dt, text


def main() -> int:
    pipe = FastCascadeS2ST(FastPipelineConfig())
    pipe._ensure_mt()

    # Warm-up: NLLB first call is dominated by model load + kernel cache init.
    print("[warmup] running 1 sequential + 1 batched pass to prime CT2 ...")
    _ = pipe.translate("xin chào.", TARGET, source="vi")
    _ = pipe.translate_sentences_batch(["xin chào."], TARGET, source="vi")

    per_utt = []
    seq_ms_all: list[float] = []
    bat_ms_all: list[float] = []
    by_count: dict[int, dict[str, list[float]]] = {}

    for utt_id, text in UTTERANCES:
        sentences = _split_sentences(text)
        n_sent = len(sentences)
        print(f"\n[utt] {utt_id}  ({n_sent} sentences)")
        print(f"      vi: {text}")

        seq_times: list[float] = []
        bat_times: list[float] = []
        seq_text = bat_text = None
        for r in range(N_REPEAT):
            dt_s, txt_s = time_sequential(pipe, sentences)
            dt_b, txt_b = time_batched(pipe, sentences)
            seq_times.append(dt_s)
            bat_times.append(dt_b)
            if seq_text is None:
                seq_text, bat_text = txt_s, txt_b

        seq_p50 = median(seq_times)
        bat_p50 = median(bat_times)
        identical = (seq_text == bat_text)
        seq_ms_all.extend(seq_times)
        bat_ms_all.extend(bat_times)
        by_count.setdefault(n_sent, {"seq": [], "bat": []})
        by_count[n_sent]["seq"].extend(seq_times)
        by_count[n_sent]["bat"].extend(bat_times)

        print(f"      seq p50 = {seq_p50:.1f} ms   bat p50 = {bat_p50:.1f} ms   "
              f"speedup = {seq_p50 / max(bat_p50, 1e-6):.2f}x   identical={identical}")
        print(f"      seq tgt: {seq_text}")
        print(f"      bat tgt: {bat_text}")
        per_utt.append({
            "utt_id": utt_id,
            "n_sentences": n_sent,
            "vi": text,
            "tgt_seq": seq_text,
            "tgt_bat": bat_text,
            "identical": identical,
            "seq_times_ms": [round(x, 2) for x in seq_times],
            "bat_times_ms": [round(x, 2) for x in bat_times],
            "seq_p50_ms": round(seq_p50, 2),
            "bat_p50_ms": round(bat_p50, 2),
            "improvement_pct": round((seq_p50 - bat_p50) / max(seq_p50, 1e-6) * 100, 2),
        })

    overall = {
        "seq_p50_ms": round(median(seq_ms_all), 2),
        "bat_p50_ms": round(median(bat_ms_all), 2),
        "improvement_pct": round(
            (median(seq_ms_all) - median(bat_ms_all)) / max(median(seq_ms_all), 1e-6) * 100, 2
        ),
        "n_samples": len(seq_ms_all),
    }
    breakdown = {}
    for n_sent, d in sorted(by_count.items()):
        seq_p = median(d["seq"])
        bat_p = median(d["bat"])
        breakdown[str(n_sent)] = {
            "n": len(d["seq"]),
            "seq_p50_ms": round(seq_p, 2),
            "bat_p50_ms": round(bat_p, 2),
            "improvement_pct": round((seq_p - bat_p) / max(seq_p, 1e-6) * 100, 2),
            "speedup": round(seq_p / max(bat_p, 1e-6), 3),
        }

    n_identical = sum(1 for u in per_utt if u["identical"])
    out = {
        "config": {
            "n_utterances":  len(UTTERANCES),
            "n_repeat":      N_REPEAT,
            "target":        TARGET,
            "mt_compute_type": pipe.cfg.mt_compute_type,
            "mt_beam":       pipe.cfg.mt_beam,
            "intra_threads": pipe.cfg.ct2_intra_threads,
        },
        "overall":     overall,
        "by_sent_count": breakdown,
        "identical_count": n_identical,
        "identical_total": len(per_utt),
        "per_utterance": per_utt,
    }
    out_path = PROJECT / "Results" / "mt_batch_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"=== overall seq p50 = {overall['seq_p50_ms']} ms   "
          f"bat p50 = {overall['bat_p50_ms']} ms   "
          f"improvement = {overall['improvement_pct']:.1f}% ===")
    print(f"[identical] {n_identical}/{len(per_utt)} utterances byte-identical")
    print(f"[write] {out_path}")
    return 0 if n_identical == len(per_utt) else 1


if __name__ == "__main__":
    sys.exit(main())
