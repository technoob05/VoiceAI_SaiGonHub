import pytest

from glossary_mt import (
    Glossary,
    GlossaryTranslator,
    Term,
    normalize,
    unused_preferred_terms,
)


@pytest.fixture(scope="module")
def gl():
    return Glossary()


def test_glossary_loads_both_tables(gl):
    assert len(gl.terms) >= 50
    assert len(gl.phrases) >= 40
    assert all(t.vi and t.en and t.zh and t.ko for t in gl.terms)


def test_injects_domain_noun(gl):
    inj = gl.inject("Xe nâng đang lùi, xin tránh ra phía sau.", "vi")
    assert "forklift" in inj.text
    assert "xe nâng" not in inj.text.casefold()
    assert [t.en for t in inj.terms] == ["forklift"]


def test_longest_term_wins(gl):
    # "máy nén khí" must not be shredded by a shorter overlapping entry.
    inj = gl.inject("Máy nén khí ở khu vực B đang quá nhiệt.", "vi")
    assert "air compressor" in inj.text
    assert "compressor" == inj.text.split("air ")[1].split()[0].strip(",.")


def test_multiple_terms_in_one_sentence(gl):
    inj = gl.inject("Anh phải đeo găng tay và kính bảo hộ trước khi vào xưởng.", "vi")
    assert "gloves" in inj.text and "safety goggles" in inj.text
    assert len(inj.terms) == 2


def test_exact_phrase_hit_needs_no_translation(gl):
    inj = gl.inject("xin chào", "vi")
    assert inj.exact_hit is not None
    assert inj.exact_hit.en == "hello"
    assert inj.text == inj.original


def test_exact_hit_ignores_trailing_punctuation(gl):
    assert gl.inject("Xin chào.", "vi").exact_hit is not None


def test_no_terms_leaves_text_untouched(gl):
    text = "Hôm nay trời rất đẹp."
    inj = gl.inject(text, "vi")
    assert inj.text == text and not inj.changed and not inj.terms


def test_english_source_is_not_injected(gl):
    text = "The forklift is reversing."
    inj = gl.inject(text, "en")
    assert inj.text == text and not inj.changed


def test_chinese_source_injects_without_word_boundaries(gl):
    inj = gl.inject("叉车正在后退，请避开。", "zh")
    assert "forklift" in inj.text


def test_korean_source_injects(gl):
    inj = gl.inject("지게차가 후진하고 있습니다.", "ko")
    assert "forklift" in inj.text


def test_unused_preferred_terms_flags_non_house_wording():
    term = Term("xe nâng", "forklift", "叉车", "지게차")
    assert unused_preferred_terms("叉车正在后退", [term], "zh") == []
    assert unused_preferred_terms("车正在后退", [term], "zh") == [term]


def test_translator_exact_hit_costs_no_mt_call(gl):
    def boom(*_args):
        raise AssertionError("translator must not be called on an exact hit")

    res = GlossaryTranslator(boom, gl).translate("xin chào", "vi", "ko")
    assert res.exact_hit and res.mt_calls == 0
    assert res.text == "안녕하세요"


def test_translator_passes_injected_text_to_model(gl):
    seen = {}

    def fake(text, source, target):
        seen["text"] = text
        return "叉车正在后退"

    res = GlossaryTranslator(fake, gl).translate(
        "Xe nâng đang lùi.", "vi", "zh")
    assert "forklift" in seen["text"]
    assert res.injected and res.mt_calls == 1
    assert res.unused_preferred == []


def test_translator_reports_when_house_term_not_used(gl):
    res = GlossaryTranslator(lambda *_: "车正在后退", gl).translate(
        "Xe nâng đang lùi.", "vi", "zh")
    assert [t.en for t in res.unused_preferred] == ["forklift"]


def test_normalize_collapses_whitespace():
    assert normalize("  a\n\t b  ") == "a b"
