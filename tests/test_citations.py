from src.citations import extract_cited_ids, is_refusal, validate_citations
from src.prompts import NO_ANSWER_MESSAGE


def test_valid_citations_are_kept():
    r = validate_citations("Leave is 27 days [1]. Remote work is 3 days [2].", [1, 2, 3])
    assert r.cited_ids == [1, 2] and r.invalid_ids == [] and r.coverage == 1.0
    assert r.clean_text == "Leave is 27 days [1]. Remote work is 3 days [2]."


def test_fabricated_ids_are_removed():
    r = validate_citations("The limit is 38 km/h [7].", [1, 2])
    assert r.invalid_ids == [7] and r.cited_ids == []
    assert "[7]" not in r.clean_text and r.clean_text == "The limit is 38 km/h."


def test_mixed_valid_and_invalid_in_one_group():
    r = validate_citations("Fact one [1, 9] and fact two [2][0].", [1, 2])
    assert r.cited_ids == [1, 2] and r.invalid_ids == [9, 0]
    assert "[9]" not in r.clean_text and "[0]" not in r.clean_text and "[1]" in r.clean_text


def test_no_valid_sources_means_every_id_invalid():
    assert validate_citations("Claim [1].", []).invalid_ids == [1]


def test_coverage_counts_only_cited_claims():
    r = validate_citations("First claim here is cited [1]. Second claim here has no citation.", [1])
    assert r.sentences_total == 2 and r.sentences_cited == 1 and r.coverage == 0.5


def test_citation_placed_after_period_still_counts():
    assert validate_citations("This claim is supported. [1]", [1]).coverage == 1.0


def test_refusal_detection_and_no_coverage():
    assert is_refusal(NO_ANSWER_MESSAGE) and not is_refusal("Leave is 27 days [1].")
    assert validate_citations(NO_ANSWER_MESSAGE, [1]).coverage is None


def test_extract_cited_ids_handles_lists():
    assert extract_cited_ids("a [1][3] b [2, 4]") == [1, 3, 2, 4]


def test_non_citation_brackets_are_untouched():
    r = validate_citations("Array syntax a[i] and [note] stay.", [1])
    assert r.clean_text == "Array syntax a[i] and [note] stay."
