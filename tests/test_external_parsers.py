import pytest

from baselines.registry import _moeim_parse, parse_seed_integers


def test_bracketed_seed_line_is_parsed_and_deduplicated():
    stdout = "loading model\nSEEDS [5, 3, 5, 9, 12, 3]\n"

    assert parse_seed_integers(stdout, budget=4) == [5, 3, 9, 12]


def test_bare_space_separated_run_is_rejected():
    with pytest.raises(ValueError, match="could not find a seed list"):
        parse_seed_integers("78 33 216 294 281 34\n", budget=3)


def test_moeim_takes_the_best_front_row_that_fits_and_deduplicates(tmp_path):
    (tmp_path / "run-1-population_default.csv").write_text(
        "n_nodes,influence,nodes\n"
        "3,120.5,\"[841, 12, 841]\"\n"
        "2,150.0,\"[7, 9]\"\n"
        "4,200.0,\"[1, 2, 3, 4]\"\n"
    )

    assert _moeim_parse(tmp_path, stdout="", budget=3) == [7, 9]
    assert _moeim_parse(tmp_path, stdout="", budget=4) == [1, 2, 3, 4]
