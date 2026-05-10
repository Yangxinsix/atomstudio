from atomstudio.paths import normalize_host_path


def test_normalize_host_path_converts_wsl_localhost_unc_path():
    assert (
        normalize_host_path(r"\\wsl.localhost\Ubuntu\home\xinyang\work\atomstudio\tests\data\water.png")
        == "/home/xinyang/work/atomstudio/tests/data/water.png"
    )


def test_normalize_host_path_converts_wsl_dollar_unc_path():
    assert normalize_host_path(r"\\wsl$\Ubuntu\home\xinyang\water.xyz") == "/home/xinyang/water.xyz"


def test_normalize_host_path_leaves_linux_path_unchanged():
    assert normalize_host_path("/home/xinyang/water.xyz") == "/home/xinyang/water.xyz"
