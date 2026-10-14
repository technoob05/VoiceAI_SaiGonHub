import json, sys, time, unicodedata
from pathlib import Path
ROOT = Path(r"d:/PhD_LetGoo/PhD_Farming/Legacy/EACL/VoiceAI_SaiGonHub")
sys.path.insert(0, str(ROOT / "prototype"))
import ctranslate2
from transformers import NllbTokenizerFast
from glossary_mt import Glossary

NLLB_CODE = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
MT_DIR = ROOT / "kaggle_uploads/voiceai-t4-models/models_ct2/nllb-200-distilled-600M-int8"

def normalize_for_scoring(text):
    text = unicodedata.normalize("NFC", text or "").casefold()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in text]
    return " ".join("".join(kept).split())

mt = ctranslate2.Translator(str(MT_DIR), device="cpu", compute_type="int8", inter_threads=1)
tok = NllbTokenizerFast.from_pretrained(str(MT_DIR), local_files_only=True)

SENTENCES = [
 "Xe nâng đang lùi, xin tránh ra phía sau.",
 "Máy nén khí ở khu vực B đang quá nhiệt, hãy dừng dây chuyền ngay.",
 "Anh phải đeo găng tay và kính bảo hộ trước khi vào xưởng.",
 "Lô hàng này cần được kiểm tra chất lượng trước khi xuất kho.",
 "Có người bị bỏng, gọi xe cứu thương ngay và báo phòng y tế.",
 "Giàn giáo chưa chắc chắn, đừng leo lên khi chưa có dây đai an toàn.",
 "Cầu trục đang nâng cốt thép, mọi người tránh xa khu vực nguy hiểm.",
 "Máy bơm bị rò rỉ khí, hãy đóng van an toàn rồi gọi tổ trưởng.",
 "Băng chuyền dừng vì hàng lỗi, quản đốc yêu cầu kiểm tra tồn kho.",
 "Trước khi vào công trường phải đội mũ bảo hộ và mang giày bảo hộ.",
 "Nhấn nút dừng khẩn cấp nếu thấy khói gần tủ điện.",
 "Ca đêm cần bốc dỡ pa lết và ghi phiếu giao hàng đầy đủ.",
]

def run(label, terms_path):
    g = Glossary(Path(terms_path))
    probes = [t.vi for t in g.terms][:24] + SENTENCES
    rows = []
    for target in ("en", "zh", "ko"):
        for probe in probes:
            def tr(text, use):
                meta = {"exact_hit": False}
                if use:
                    p = g.inject(text, "vi")
                    if p.exact_hit is not None:
                        return p.exact_hit.get(target), {"exact_hit": True}
                    text = p.text
                tok.src_lang = NLLB_CODE["vi"]
                toks = tok.convert_ids_to_tokens(tok.encode(text))
                r = mt.translate_batch([toks], target_prefix=[[NLLB_CODE[target]]], beam_size=1,
                                       max_decoding_length=min(256, max(16, len(toks) * 2 + 4)))
                return tok.decode(tok.convert_tokens_to_ids(r[0].hypotheses[0][1:]),
                                  skip_special_tokens=True).strip(), meta
            off, _ = tr(probe, False)
            on, meta = tr(probe, True)
            expected = list(g.find_terms(probe, "vi"))
            has = lambda t, o: normalize_for_scoring(t) in normalize_for_scoring(o)
            rows.append({"target": target, "source_text": probe, "off": off, "on": on,
                         "exact_hit": meta["exact_hit"],
                         "expected_terms": [t.get(target) for t in expected],
                         "recovered_off": sum(has(t.get(target), off) for t in expected),
                         "recovered_on": sum(has(t.get(target), on) for t in expected),
                         "n_expected": len(expected)})
        print(f"  {label} {target} done", flush=True)
    wt = [r for r in rows if r["n_expected"] > 0]
    sent = [r for r in wt if not r["exact_hit"]]
    ex = [r for r in wt if r["exact_hit"]]
    def rec(rs):
        e = sum(r["n_expected"] for r in rs)
        if not e: return {"n_terms": 0}
        return {"n_probes": len(rs), "n_terms": e,
                "off": sum(r["recovered_off"] for r in rs) / e,
                "on": sum(r["recovered_on"] for r in rs) / e}
    return {"label": label, "n_glossary_terms": len(g.terms), "n_probes": len(rows),
            "translated_sentences": rec(sent), "exact_hits": rec(ex), "all": rec(wt),
            "rows": rows}

out = {}
for label, path in [("old61", ROOT / "Results/kaggle_t4/_terms_old61.json"),
                    ("new93", ROOT / "prototype/industrial_terms.json")]:
    t0 = time.time()
    out[label] = run(label, path)
    print(label, json.dumps(out[label]["translated_sentences"]), f"{time.time()-t0:.0f}s", flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
