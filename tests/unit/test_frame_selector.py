from atomstudio.io.ase_loader import parse_frame_selector


def test_parse_all():
    assert parse_frame_selector("all", 4) == [0, 1, 2, 3]


def test_parse_last():
    assert parse_frame_selector("last", 4) == [3]


def test_parse_slice():
    assert parse_frame_selector("1:5:2", 6) == [1, 3]


def test_parse_index_negative():
    assert parse_frame_selector("-1", 4) == [3]

