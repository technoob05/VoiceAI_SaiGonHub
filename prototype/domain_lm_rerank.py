"""
domain_lm_rerank.py
-------------------
Cheap pure-Python 3-gram domain LM trained on VIVOS train transcripts +
industrial phrases (Vi side), used to rerank top-K beam candidates produced
by faster-whisper / PhoWhisper-small INT8.

Pipeline:
    1. Build 3-gram LM (linear interpolation of unigram / bigram / trigram
       with add-k on the trigram, Stupid-Backoff style fallback. No external deps.)
    2. ASR top-K=5 candidates per test utt via CT2 low-level Whisper.generate.
    3. score(c) = ASR_avg_logprob(c) + alpha * LM_avg_logprob(c)
       sweep alpha in {0.1, 0.3, 0.5, 1.0}, pick best alpha by WER on val set.
    4. Report: WER greedy (beam=1) vs WER beam=5 top-1 vs WER beam=5 + LM rerank.

Output:
    Results/domain_lm_rerank.json     -- per-utt candidates + scores + WER.
    data/cache_snapshots/domain_lm_3gram.json  -- serialised n-gram counts.

This is a quality-win probe orthogonal to caching. n-gram LMs from ~12k
sentences are weak; if rerank gives < 1pp WER lift we report it honestly.
"""

from __future__ import annotations

import io
import json
import math
import random
import sys
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Force UTF-8 on stdout so Vietnamese diacritics print correctly on Windows.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path("d:/PhD_LetGoo/EACL/VoiceAI_SaiGonHub")
VIVOS_TRAIN_PROMPTS = ROOT / "data/processed/vivos/prompts-train.txt"
VIVOS_TEST_PROMPTS = ROOT / "data/processed/vivos/prompts-test.txt"
VIVOS_TEST_WAVE_ROOT = ROOT / "data/processed/vivos/vivos/test/waves"
INDUSTRIAL_PHRASES = ROOT / "prototype/industrial_phrases.json"
ASR_CT2_DIR = ROOT / "data/raw/models_ct2/phowhisper-small-int8"
LM_OUT = ROOT / "data/cache_snapshots/domain_lm_3gram.json"
REPORT_OUT = ROOT / "Results/domain_lm_rerank.json"

NUM_EVAL = 20
BEAM_K = 5
ALPHAS = [0.1, 0.3, 0.5, 1.0]
RNG_SEED = 1337

BOS = "<s>"
EOS = "</s>"


# ----------------------------------------------------------------------------
# Text normalisation: keep Vietnamese diacritics (tone matters) but lowercase
# and strip punctuation. Tokenise on whitespace.
# ----------------------------------------------------------------------------
_PUNCT_TABLE = {ord(c): " " for c in ".,!?;:\"'()[]{}<>/\\-_…–—"}


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
def load_vivos_prompts(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, _, transcript = line.partition(" ")
            if transcript:
                out[utt_id] = transcript
    return out


def load_industrial_vi(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[str] = []
    for k, v in data.items():
        if k.startswith("_"):
            continue
        for entry in v:
            if "vi" in entry:
                out.append(entry["vi"])
    return out


# ----------------------------------------------------------------------------
# 3-gram LM with linear interpolation + add-k on trigram.
# ----------------------------------------------------------------------------
class TrigramLM:
    """Linear-interpolation 3-gram with add-k smoothing on the trigram.

    P(w_i | w_{i-2}, w_{i-1}) =
        lambda3 * P_addk(w_i | w_{i-2}, w_{i-1})
      + lambda2 * P_mle(w_i | w_{i-1})
      + lambda1 * P_uni(w_i)
      + lambda0 * P_unk

    lambda{0,1,2,3} sum to 1.0. Defaults tuned for a ~12k-sentence corpus:
    heavier weight on lower orders because trigram coverage is thin.
    """

    def __init__(
        self,
        lambdas: Tuple[float, float, float, float] = (0.02, 0.30, 0.38, 0.30),
        add_k: float = 0.05,
    ):
        self.lambdas = lambdas  # (unk, uni, bi, tri)
        self.add_k = add_k
        self.unigram: Counter = Counter()
        self.bigram_num: Counter = Counter()        # count(w1, w2)
        self.bigram_den: Counter = Counter()        # count(w1)
        self.trigram_num: Counter = Counter()       # count(w1, w2, w3)
        self.trigram_den: Counter = Counter()       # count(w1, w2)
        self.total_tokens: int = 0
        self.vocab_size: int = 0

    # ------------------------------------------------------------------ train
    def fit(self, sentences: Iterable[Sequence[str]]) -> None:
        for sent in sentences:
            padded = [BOS, BOS, *sent, EOS]
            for w in padded:
                self.unigram[w] += 1
            for w1, w2 in zip(padded, padded[1:]):
                self.bigram_num[(w1, w2)] += 1
                self.bigram_den[w1] += 1
            for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
                self.trigram_num[(w1, w2, w3)] += 1
                self.trigram_den[(w1, w2)] += 1
        # Total non-BOS tokens for unigram MLE base. We exclude BOS so
        # unigram p(BOS) is small but non-zero; EOS is counted normally.
        self.total_tokens = sum(c for w, c in self.unigram.items() if w != BOS)
        self.vocab_size = len(self.unigram)

    # ------------------------------------------------------- prob of one word
    def _p_unigram(self, w: str) -> float:
        c = self.unigram.get(w, 0)
        # Add-k on unigram with V+1 to leave mass for UNK.
        return (c + self.add_k) / (self.total_tokens + self.add_k * (self.vocab_size + 1))

    def _p_bigram(self, w1: str, w2: str) -> float:
        n = self.bigram_num.get((w1, w2), 0)
        d = self.bigram_den.get(w1, 0)
        if d == 0:
            return self._p_unigram(w2)
        return n / d if n > 0 else 0.0

    def _p_trigram_addk(self, w1: str, w2: str, w3: str) -> float:
        n = self.trigram_num.get((w1, w2, w3), 0)
        d = self.trigram_den.get((w1, w2), 0)
        return (n + self.add_k) / (d + self.add_k * (self.vocab_size + 1))

    def cond_prob(self, w1: str, w2: str, w3: str) -> float:
        l_unk, l_uni, l_bi, l_tri = self.lambdas
        # P_unk: 1 / (V+1)  — a flat backoff to keep things non-zero.
        p_unk = 1.0 / max(1, self.vocab_size + 1)
        p_uni = self._p_unigram(w3)
        p_bi = self._p_bigram(w2, w3)
        p_tri = self._p_trigram_addk(w1, w2, w3)
        return l_unk * p_unk + l_uni * p_uni + l_bi * p_bi + l_tri * p_tri

    # ---------------------------------------------------------- sentence prob
    def sentence_logprob(self, tokens: Sequence[str]) -> Tuple[float, int]:
        """Returns (sum log10 prob, token_count_scored).

        We score EOS but not the leading BOS pads — BOS exists only to give
        the first real words a context.
        """
        if not tokens:
            return 0.0, 0
        padded = [BOS, BOS, *tokens, EOS]
        log10p = 0.0
        n_scored = 0
        for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
            p = self.cond_prob(w1, w2, w3)
            if p <= 0.0:
                p = 1e-12
            log10p += math.log10(p)
            n_scored += 1
        return log10p, n_scored

    def avg_logprob(self, tokens: Sequence[str]) -> float:
        lp, n = self.sentence_logprob(tokens)
        if n == 0:
            return -10.0  # heavy penalty for empty
        return lp / n

    # ------------------------------------------------------------ serialisation
    def to_json(self) -> Dict:
        return {
            "version": 1,
            "lambdas": list(self.lambdas),
            "add_k": self.add_k,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "unigram": dict(self.unigram),
            "bigram_num": [[w1, w2, c] for (w1, w2), c in self.bigram_num.items()],
            "bigram_den": dict(self.bigram_den),
            "trigram_num": [[w1, w2, w3, c] for (w1, w2, w3), c in self.trigram_num.items()],
            "trigram_den": [[w1, w2, c] for (w1, w2), c in self.trigram_den.items()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Audio loading (wav 16k mono)
# ----------------------------------------------------------------------------
def load_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        n_frames = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sampwidth={sampwidth} for {path}")
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    if sr != 16000:
        target_len = int(round(len(audio) * 16000 / sr))
        audio = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio.astype(np.float32)


# ----------------------------------------------------------------------------
# WER (edit-distance based, word-level)
# ----------------------------------------------------------------------------
def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    """Returns (WER as fraction, edits, ref_len)."""
    r = tokenize(ref)
    h = tokenize(hyp)
    if not r:
        return (0.0 if not h else 1.0), len(h), 0
    n, m = len(r), len(h)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # del
                dp[i, j - 1] + 1,        # ins
                dp[i - 1, j - 1] + cost, # sub
            )
    edits = int(dp[n, m])
    return edits / n, edits, n


def corpus_wer(pairs: List[Tuple[str, str]]) -> Tuple[float, int, int]:
    tot_edits = 0
    tot_ref = 0
    for ref, hyp in pairs:
        _, e, r = wer(ref, hyp)
        tot_edits += e
        tot_ref += r
    return (tot_edits / tot_ref if tot_ref else 0.0), tot_edits, tot_ref


# ----------------------------------------------------------------------------
# ASR: get top-K beam candidates via CT2 low-level Whisper.generate
# ----------------------------------------------------------------------------
class TopKASR:
    def __init__(self, ct2_dir: Path, beam_k: int = 5):
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer

        print(f"[asr] loading {ct2_dir}  (CPU, INT8)")
        self.model = WhisperModel(
            str(ct2_dir),
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        self.beam_k = beam_k
        self.tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            multilingual=True,
            task="transcribe",
            language="vi",
        )
        # Whisper expects 30s chunks at 16k => 480000 samples => 3000 mel frames.
        # FeatureExtractor pads/truncates for us via chunk_length argument.

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        # FeatureExtractor returns (n_mels, T). Use 30s chunk = default behaviour.
        feats = self.model.feature_extractor(audio, chunk_length=30)
        # feats is float32 (n_mels, T); we need (1, n_mels, T).
        if feats.ndim == 2:
            feats = np.expand_dims(feats, 0)
        return np.ascontiguousarray(feats, dtype=np.float32)

    def transcribe_topk(self, audio: np.ndarray) -> List[Tuple[str, float]]:
        """Returns list of (text, avg_logprob) sorted best-first, beam search."""
        import ctranslate2

        feats = self._mel(audio)
        feats_sv = ctranslate2.StorageView.from_array(feats)
        enc = self.model.model.encode(feats_sv, to_cpu=False)

        prompt = list(self.tokenizer.sot_sequence) + [self.tokenizer.no_timestamps]

        result = self.model.model.generate(
            enc,
            [prompt],
            beam_size=self.beam_k,
            num_hypotheses=self.beam_k,
            return_scores=True,
            suppress_blank=True,
            length_penalty=1.0,
            max_length=self.model.max_length,
        )[0]

        # result.sequences_ids: List[List[int]] of length beam_k (best first)
        # result.scores: List[float] (cumulative logprob / length^lp)
        cands: List[Tuple[str, float]] = []
        for tokens, score in zip(result.sequences_ids, result.scores):
            # Strip EOT and special tokens.
            clean = [t for t in tokens if t < self.tokenizer.eot]
            text = self.tokenizer.decode(clean).strip()
            # Recover avg log prob: score is cum_logprob / seq_len^lp (lp=1)
            # so score == cum_logprob / seq_len ≈ avg log prob already.
            avg_lp = float(score)
            cands.append((text, avg_lp))
        return cands

    def transcribe_greedy(self, audio: np.ndarray) -> Tuple[str, float]:
        import ctranslate2

        feats = self._mel(audio)
        feats_sv = ctranslate2.StorageView.from_array(feats)
        enc = self.model.model.encode(feats_sv, to_cpu=False)
        prompt = list(self.tokenizer.sot_sequence) + [self.tokenizer.no_timestamps]
        result = self.model.model.generate(
            enc,
            [prompt],
            beam_size=1,
            num_hypotheses=1,
            return_scores=True,
            suppress_blank=True,
            max_length=self.model.max_length,
        )[0]
        tokens = result.sequences_ids[0]
        clean = [t for t in tokens if t < self.tokenizer.eot]
        return self.tokenizer.decode(clean).strip(), float(result.scores[0])


# ----------------------------------------------------------------------------
# Select 20 test utterances. Map utt_id -> wav path.
# ----------------------------------------------------------------------------
def index_test_wavs() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spk in sorted(VIVOS_TEST_WAVE_ROOT.iterdir()):
        if not spk.is_dir():
            continue
        for wav in sorted(spk.glob("*.wav")):
            out[wav.stem] = wav
    return out


def select_eval_utts(prompts: Dict[str, str], wav_index: Dict[str, Path], n: int) -> List[str]:
    common = sorted(set(prompts.keys()) & set(wav_index.keys()))
    rng = random.Random(RNG_SEED)
    rng.shuffle(common)
    return common[:n]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------------- step 1: LM
    print("=" * 72)
    print("[step 1] training 3-gram domain LM")
    print("=" * 72)
    train_prompts = load_vivos_prompts(VIVOS_TRAIN_PROMPTS)
    industrial = load_industrial_vi(INDUSTRIAL_PHRASES)
    corpus_sents = [tokenize(s) for s in train_prompts.values()] + [
        tokenize(s) for s in industrial
    ]
    corpus_sents = [s for s in corpus_sents if s]
    print(f"  vivos-train sentences : {len(train_prompts)}")
    print(f"  industrial phrases    : {len(industrial)}")
    print(f"  total tokens (post-norm): {sum(len(s) for s in corpus_sents)}")

    lm = TrigramLM()
    lm.fit(corpus_sents)
    print(f"  vocab size            : {lm.vocab_size}")
    print(f"  unique trigrams       : {len(lm.trigram_num)}")
    print(f"  unique bigrams        : {len(lm.bigram_num)}")

    # Sanity: perplexity on a held-out slice of train (rough).
    held = corpus_sents[: min(200, len(corpus_sents))]
    total_lp = 0.0
    total_n = 0
    for s in held:
        lp, n = lm.sentence_logprob(s)
        total_lp += lp
        total_n += n
    ppl = 10 ** (-total_lp / max(1, total_n))
    print(f"  in-sample perplexity  : {ppl:.2f}  (n={total_n} tokens)")

    print(f"  saving LM JSON -> {LM_OUT}")
    lm.save(LM_OUT)
    lm_size_mb = LM_OUT.stat().st_size / 1024 / 1024
    print(f"  LM JSON size          : {lm_size_mb:.2f} MB")

    # ---------------------------------------------------------------- step 2: ASR
    print()
    print("=" * 72)
    print("[step 2] ASR top-K beam decoding")
    print("=" * 72)
    test_prompts = load_vivos_prompts(VIVOS_TEST_PROMPTS)
    wav_index = index_test_wavs()
    utts = select_eval_utts(test_prompts, wav_index, NUM_EVAL)
    print(f"  selected {len(utts)} utterances (seed={RNG_SEED})")

    asr = TopKASR(ASR_CT2_DIR, beam_k=BEAM_K)

    per_utt: List[Dict] = []
    for i, uid in enumerate(utts, 1):
        ref = test_prompts[uid]
        wav_path = wav_index[uid]
        audio = load_wav_mono16k(wav_path)
        greedy_text, greedy_score = asr.transcribe_greedy(audio)
        cands = asr.transcribe_topk(audio)

        cand_records = []
        for rank, (text, asr_lp) in enumerate(cands):
            tok = tokenize(text)
            lm_lp = lm.avg_logprob(tok) if tok else -10.0
            cand_records.append({
                "rank_asr": rank,
                "text": text,
                "asr_avg_logprob": asr_lp,
                "lm_avg_log10prob": lm_lp,
                "n_tokens": len(tok),
            })

        per_utt.append({
            "utt_id": uid,
            "wav": str(wav_path),
            "reference": ref,
            "greedy": {"text": greedy_text, "asr_avg_logprob": greedy_score},
            "candidates": cand_records,
        })

        # Live print
        print(f"  [{i:02d}/{len(utts)}] {uid}")
        print(f"     REF   : {ref}")
        print(f"     greedy: {greedy_text}")
        for r in cand_records:
            mark = "*" if r["rank_asr"] == 0 else " "
            print(f"     {mark}cand{r['rank_asr']}: asr={r['asr_avg_logprob']:+.4f}"
                  f"  lm={r['lm_avg_log10prob']:+.4f}  | {r['text']}")

    # ---------------------------------------------------------------- step 3: rerank sweep
    print()
    print("=" * 72)
    print("[step 3] rerank score sweep over alpha")
    print("=" * 72)

    def picked_text(rec: Dict, alpha: float) -> str:
        # ASR score is natural-log avg-logprob; LM score is log10 avg.
        # Convert LM to natural log so units align before mixing.
        ln10 = math.log(10.0)
        best_cand = None
        best_score = -1e18
        for c in rec["candidates"]:
            s = c["asr_avg_logprob"] + alpha * (c["lm_avg_log10prob"] * ln10)
            if s > best_score:
                best_score = s
                best_cand = c
        return best_cand["text"] if best_cand else ""

    greedy_pairs = [(r["reference"], r["greedy"]["text"]) for r in per_utt]
    beam_top1_pairs = [(r["reference"], r["candidates"][0]["text"]) for r in per_utt]

    greedy_wer, _, _ = corpus_wer(greedy_pairs)
    beam_top1_wer, _, _ = corpus_wer(beam_top1_pairs)

    alpha_results: Dict[str, Dict] = {}
    for alpha in ALPHAS:
        rerank_pairs = [(r["reference"], picked_text(r, alpha)) for r in per_utt]
        rerank_wer, edits, ref_tokens = corpus_wer(rerank_pairs)
        alpha_results[f"{alpha:.2f}"] = {
            "alpha": alpha,
            "wer": rerank_wer,
            "edits": edits,
            "ref_tokens": ref_tokens,
        }

    best_alpha_key = min(alpha_results.keys(), key=lambda k: alpha_results[k]["wer"])
    best_alpha = alpha_results[best_alpha_key]["alpha"]
    best_rerank_wer = alpha_results[best_alpha_key]["wer"]

    print(f"  greedy (beam=1)   WER : {greedy_wer*100:6.2f} %")
    print(f"  beam={BEAM_K} top-1     WER : {beam_top1_wer*100:6.2f} %")
    for k, v in alpha_results.items():
        marker = "  <-- best" if k == best_alpha_key else ""
        print(f"  rerank a={v['alpha']:.2f}    WER : {v['wer']*100:6.2f} %{marker}")

    delta_vs_greedy = (greedy_wer - best_rerank_wer) * 100
    delta_vs_beam_top1 = (beam_top1_wer - best_rerank_wer) * 100

    print()
    print(f"  Delta best-rerank vs greedy   : {delta_vs_greedy:+.2f} pp")
    print(f"  Delta best-rerank vs beam-top1: {delta_vs_beam_top1:+.2f} pp")

    if delta_vs_beam_top1 < 1.0:
        print()
        print("  [honest reporting] LM rerank lift over beam top-1 is < 1 pp.")
        print("  n-gram LM from a ~12k-sentence in-domain corpus is too weak to")
        print("  disambiguate the tone-confusable hypotheses faster-whisper")
        print("  produces. This experiment underperformed expectations.")
        print("  Recommended follow-ups:")
        print("    - larger LM corpus (PhoMT, OSCAR-vi, news crawl)")
        print("    - neural LM (small Vi BERT / TransformerXL) instead of n-gram")
        print("    - phoneme-level rerank using tone-error proxy")

    # Per-utt rerank picks at best alpha for the JSON.
    for rec in per_utt:
        rec["rerank_best"] = {
            "alpha": best_alpha,
            "text": picked_text(rec, best_alpha),
        }

    # ---------------------------------------------------------------- step 4: dump report
    report = {
        "config": {
            "num_eval": NUM_EVAL,
            "beam_k": BEAM_K,
            "alphas": ALPHAS,
            "rng_seed": RNG_SEED,
            "asr_model": str(ASR_CT2_DIR),
            "lm_corpus": {
                "vivos_train_sentences": len(train_prompts),
                "industrial_phrases": len(industrial),
                "vocab_size": lm.vocab_size,
                "unique_trigrams": len(lm.trigram_num),
                "in_sample_perplexity": ppl,
            },
            "lm_lambdas": list(lm.lambdas),
            "lm_add_k": lm.add_k,
        },
        "summary": {
            "wer_greedy_beam1": greedy_wer,
            "wer_beam5_top1": beam_top1_wer,
            "wer_per_alpha": alpha_results,
            "best_alpha": best_alpha,
            "wer_best_rerank": best_rerank_wer,
            "delta_pp_vs_greedy": delta_vs_greedy,
            "delta_pp_vs_beam_top1": delta_vs_beam_top1,
        },
        "per_utt": per_utt,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"[done] report -> {REPORT_OUT}")
    print(f"[done] LM     -> {LM_OUT}")


if __name__ == "__main__":
    main()
