"""Conservative Vietnamese transcript correction for the offline edge pipeline.

The neural model runs only on an immutable ASR final. Its output is accepted only
when it preserves numbers and protected terms and stays within a bounded edit
distance. A rejected or unavailable model is a normal fallback, never a pipeline
failure.
"""

from __future__ import annotations

import os
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional


DEFAULT_MODEL = "bmd1905/vietnamese-correction-v2"
DEFAULT_CT2_MODEL = Path(__file__).resolve().parents[1] / "data" / "raw" / "models_ct2" / "vietnamese-correction-v2-int8"
_WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
_NEGATIONS = {"không", "chẳng", "chưa", "đừng", "chớ"}
_ACRONYM_RE = re.compile(r"(?<!\w)[A-ZĐ][A-ZĐ0-9-]{1,}(?!\w)")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _WORD_RE.findall(text)]


def _skeleton(token: str) -> str:
    """Token stripped of diacritics — a respelling of the same word, not a new one."""
    decomposed = unicodedata.normalize("NFD", token.replace("đ", "d"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _dropped_tokens(raw_tokens: list[str], proposed_tokens: list[str]) -> list[str]:
    """Words the corrector deleted outright rather than respelled.

    A spelling corrector may rewrite a word; it must not silently remove one.
    Deleting "ủy" from "ủy ban châu âu" costs the translator a whole entity
    while staying well inside the length and edit-ratio budgets, so it needs a
    guard of its own.
    """
    dropped: list[str] = []
    matcher = SequenceMatcher(None, raw_tokens, proposed_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            dropped.extend(raw_tokens[i1:i2])
        elif tag == "replace":
            removed, added = raw_tokens[i1:i2], proposed_tokens[j1:j2]
            surplus = len(removed) - len(added)
            if surplus <= 0:
                continue
            unmatched = [
                token for token in removed
                if not any(_skeleton(token) == _skeleton(other) for other in added)
            ]
            dropped.extend(unmatched[:surplus] or removed[:surplus])
    return dropped


@dataclass
class CorrectorConfig:
    model: str = os.environ.get("VOICEAI_CORRECTOR_MODEL", DEFAULT_MODEL).strip()
    local_files_only: bool = _env_bool("VOICEAI_CORRECTOR_LOCAL_ONLY", True)
    dynamic_int8: bool = _env_bool("VOICEAI_CORRECTOR_INT8", False)
    ct2_model: str = os.environ.get("VOICEAI_CORRECTOR_CT2", str(DEFAULT_CT2_MODEL)).strip()
    max_input_tokens: int = int(os.environ.get("VOICEAI_CORRECTOR_MAX_INPUT", "256"))
    max_new_tokens: int = int(os.environ.get("VOICEAI_CORRECTOR_MAX_NEW", "256"))
    max_edit_ratio: float = float(os.environ.get("VOICEAI_CORRECTOR_MAX_EDIT", "0.35"))
    max_dropped_tokens: int = int(os.environ.get("VOICEAI_CORRECTOR_MAX_DROPPED", "0"))
    min_length_ratio: float = float(os.environ.get("VOICEAI_CORRECTOR_MIN_LENGTH", "0.72"))
    max_length_ratio: float = float(os.environ.get("VOICEAI_CORRECTOR_MAX_LENGTH", "1.35"))
    protected_terms: tuple[str, ...] = field(default_factory=lambda: tuple(
        term.strip() for term in os.environ.get(
            "VOICEAI_PROTECTED_TERMS",
            "AI,NPU,QNN,QAIRT,CPU,GPU,USB,Wi-Fi,Bluetooth,PhoWhisper,NLLB",
        ).split(",") if term.strip()
    ))


@dataclass
class CorrectionResult:
    raw_text: str
    proposed_text: str
    text: str
    accepted: bool
    reason: str
    edit_ratio: float
    latency_ms: float
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptCorrector:
    def __init__(
        self,
        config: CorrectorConfig | None = None,
        generator: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or CorrectorConfig()
        self._generator = generator
        self._tokenizer = None
        self._model = None
        self._ct2 = None
        self._load_error: Optional[str] = None

    def _ensure(self) -> None:
        if self._generator is not None or self._load_error is not None:
            return
        try:
            ct2_path = Path(self.config.ct2_model)
            if (ct2_path / "model.bin").is_file():
                self._load_ct2(ct2_path)
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            kwargs = {"local_files_only": self.config.local_files_only}
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model,
                    use_safetensors=True,
                    **kwargs,
                ).eval()
            except Exception as exc:
                # transformers 5.0.0 shipped MBartConfig.scale_embedding with a
                # bool default even though strict validation declares it as int.
                # Older mBART checkpoints therefore fail before weights load.
                if type(exc).__name__ != "StrictDataclassFieldValidationError":
                    raise
                model = self._load_mbart_transformers5_compat().eval()
            if self.config.dynamic_int8:
                try:
                    quantize = getattr(torch.ao.quantization, "quantize_dynamic", None)
                    if quantize is not None:
                        model = quantize(model, {torch.nn.Linear}, dtype=torch.qint8)
                except Exception as exc:
                    print(f"[warn] corrector INT8 unavailable ({exc}); using FP32")
            self._model = model
        except Exception as exc:
            self._load_error = str(exc)

    def _load_ct2(self, model_path: Path) -> None:
        import ctranslate2
        from transformers import BartphoTokenizer

        self._tokenizer = BartphoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._ct2 = ctranslate2.Translator(
            str(model_path), device="cpu", compute_type="int8",
            intra_threads=max((os.cpu_count() or 4) - 2, 4), inter_threads=1,
        )

    def _load_mbart_transformers5_compat(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSeq2SeqLM, BartphoTokenizer, MBartConfig

        source = Path(self.config.model)
        if not source.exists():
            source = Path(snapshot_download(
                repo_id=self.config.model,
                local_files_only=self.config.local_files_only,
            ))

        config_data = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if config_data.get("model_type") != "mbart":
            raise RuntimeError("strict config failure is not an mBART checkpoint")

        for item in fields(MBartConfig):
            if item.name == "scale_embedding" and isinstance(item.default, bool):
                item.default = int(item.default)
        if isinstance(config_data.get("scale_embedding"), bool):
            config_data["scale_embedding"] = int(config_data["scale_embedding"])

        config = MBartConfig.from_dict(config_data)
        # This checkpoint is mBART-shaped but uses VinAI's 40k BARTpho
        # vocabulary. MBartTokenizer would produce ids up to 250k and crash the
        # 40,030-row embedding table.
        self._tokenizer = BartphoTokenizer.from_pretrained(
            str(source), local_files_only=True
        )
        return AutoModelForSeq2SeqLM.from_pretrained(
            str(source),
            config=config,
            use_safetensors=True,
            local_files_only=True,
        )

    def prewarm(self) -> bool:
        self._ensure()
        return self._generator is not None or self._model is not None or self._ct2 is not None

    def _generate(self, text: str) -> str:
        if self._generator is not None:
            return self._generator(text)
        self._ensure()
        if self._tokenizer is None:
            raise RuntimeError(self._load_error or "corrector model is unavailable")

        if self._ct2 is not None:
            input_ids = self._tokenizer(text, add_special_tokens=True)["input_ids"]
            source_tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
            result = self._ct2.translate_batch(
                [source_tokens], beam_size=1,
                max_decoding_length=min(
                    self.config.max_new_tokens, max(len(source_tokens) + 16, 24)
                ),
                repetition_penalty=1.05,
                no_repeat_ngram_size=4,
            )[0]
            output_ids = self._tokenizer.convert_tokens_to_ids(result.hypotheses[0])
            return self._tokenizer.decode(output_ids, skip_special_tokens=True)

        if self._model is None:
            raise RuntimeError(self._load_error or "corrector model is unavailable")

        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.05,
                no_repeat_ngram_size=4,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _validate(self, raw: str, proposed: str) -> tuple[bool, str, float]:
        if not proposed:
            return False, "empty_output", 1.0
        if raw.casefold() == proposed.casefold():
            return True, "unchanged", 0.0

        length_ratio = len(proposed) / max(len(raw), 1)
        if not self.config.min_length_ratio <= length_ratio <= self.config.max_length_ratio:
            return False, "length_guard", 1.0

        raw_numbers = _NUMBER_RE.findall(raw)
        proposed_numbers = _NUMBER_RE.findall(proposed)
        if raw_numbers != proposed_numbers:
            return False, "number_guard", 1.0

        raw_negations = [token for token in _tokens(raw) if token in _NEGATIONS]
        proposed_negations = [token for token in _tokens(proposed) if token in _NEGATIONS]
        if raw_negations != proposed_negations:
            return False, "negation_guard", 1.0

        proposed_fold = proposed.casefold()
        for term in self.config.protected_terms:
            if term.casefold() in raw.casefold() and term.casefold() not in proposed_fold:
                return False, f"protected_term:{term}", 1.0

        if _ACRONYM_RE.findall(raw) != _ACRONYM_RE.findall(proposed):
            return False, "acronym_guard", 1.0

        raw_tokens = _tokens(raw)
        proposed_tokens = _tokens(proposed)
        edit_ratio = 1.0 - SequenceMatcher(None, raw_tokens, proposed_tokens).ratio()
        if edit_ratio > self.config.max_edit_ratio:
            return False, "edit_guard", edit_ratio

        dropped = _dropped_tokens(raw_tokens, proposed_tokens)
        if len(dropped) > self.config.max_dropped_tokens:
            return False, f"deletion_guard:{','.join(dropped[:3])}", edit_ratio
        return True, "accepted", edit_ratio

    def correct(self, text: str) -> CorrectionResult:
        raw = normalize_text(text)
        if not raw:
            return CorrectionResult(raw, raw, raw, False, "empty_input", 0.0, 0.0, self.config.model)

        started = time.perf_counter()
        try:
            proposed = normalize_text(self._generate(raw))
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            reason = "model_unavailable" if self._load_error else f"model_error:{type(exc).__name__}"
            return CorrectionResult(raw, "", raw, False, reason, 0.0, latency, self.config.model)

        accepted, reason, edit_ratio = self._validate(raw, proposed)
        latency = (time.perf_counter() - started) * 1000
        return CorrectionResult(
            raw_text=raw,
            proposed_text=proposed,
            text=proposed if accepted else raw,
            accepted=accepted,
            reason=reason,
            edit_ratio=edit_ratio,
            latency_ms=latency,
            model=self.config.model,
        )


def model_cache_path(project_root: Path, model: str = DEFAULT_MODEL) -> Path:
    slug = model.replace("/", "--")
    return project_root / "data" / "raw" / "hf_cache" / "hub" / f"models--{slug}"
