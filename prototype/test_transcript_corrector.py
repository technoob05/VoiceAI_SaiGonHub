from transcript_corrector import CorrectorConfig, TranscriptCorrector


def make_corrector(output: str, **kwargs) -> TranscriptCorrector:
    config = CorrectorConfig(dynamic_int8=False, **kwargs)
    return TranscriptCorrector(config, generator=lambda _text: output)


def test_accepts_conservative_contextual_correction():
    result = make_corrector("Tôi đang làm việc trong nhà máy.").correct(
        "tôi đang làm viêc trong nhà máy"
    )
    assert result.accepted
    assert result.text == "Tôi đang làm việc trong nhà máy."


def test_rejects_number_mutation():
    result = make_corrector("Đặt áp suất ở mức 50 bar.").correct(
        "đặt áp suất ở mức 15 bar"
    )
    assert not result.accepted
    assert result.reason == "number_guard"
    assert result.text == "đặt áp suất ở mức 15 bar"


def test_rejects_protected_term_mutation():
    result = make_corrector("Khởi động CPU ngay.").correct("khởi động NPU ngay")
    assert not result.accepted
    assert result.reason == "protected_term:NPU"


def test_rejects_semantic_rewrite():
    result = make_corrector(
        "Hôm nay trời đẹp nên chúng ta đi dạo ngoài công viên.",
        max_edit_ratio=0.2,
    ).correct("dừng máy ngay lập tức")
    assert not result.accepted
    assert result.text == "dừng máy ngay lập tức"


def test_model_failure_falls_back_to_raw():
    def fail(_text: str) -> str:
        raise RuntimeError("offline checkpoint missing")

    result = TranscriptCorrector(CorrectorConfig(), generator=fail).correct("xin chào")
    assert not result.accepted
    assert result.text == "xin chào"
    assert result.reason == "model_error:RuntimeError"


def test_rejects_negation_mutation():
    raw = "Tôi không muốn tiếp tục vận hành máy."
    result = make_corrector("Tôi muốn tiếp tục vận hành máy.").correct(raw)
    assert not result.accepted
    assert result.text == raw
    assert result.reason == "negation_guard"


def test_rejects_unknown_acronym_mutation():
    raw = "Thiết bị XYZ đã sẵn sàng."
    result = make_corrector("Thiết bị ABC đã sẵn sàng.").correct(raw)
    assert not result.accepted
    assert result.text == raw
    assert result.reason == "acronym_guard"


def test_rejects_word_deletion():
    raw = "ủy ban châu âu kêu gọi những quốc gia phát triển còn lại."
    result = make_corrector("ban châu Âu kêu gọi những quốc gia phát triển còn lại.").correct(raw)
    assert not result.accepted
    assert result.reason.startswith("deletion_guard")
    assert result.text == raw


def test_rejects_entity_truncation():
    raw = "y tế bình hưng hòa bị hư hỏng."
    result = make_corrector("y tế bình hưng bị hư hỏng.").correct(raw)
    assert not result.accepted
    assert result.reason.startswith("deletion_guard")


def test_allows_diacritic_respelling():
    result = make_corrector("mỗi khi xe lát đi ngang qua thắng kẽn mới đem ra đổ.").correct(
        "mỗi khi xe lát đi ngang qua tắng kẽn mới đem ra đổ."
    )
    assert result.accepted
    assert result.reason == "accepted"
