from vam_timeline_ai.motion.windows import make_default_window_set, make_windows, window_id


def test_make_windows_overlapping():
    assert make_windows(5.0, 2.0, 1.0) == [
        (0.0, 2.0),
        (1.0, 3.0),
        (2.0, 4.0),
        (3.0, 5.0),
    ]


def test_make_windows_keeps_short_sources():
    assert make_windows(1.25, 2.0, 1.0) == [(0.0, 1.25)]


def test_default_window_set_contains_2_4_8_second_groups():
    windows = make_default_window_set(10.0)

    assert set(windows) == {"2s_stride_1s", "4s_stride_2s", "8s_stride_4s"}
    assert windows["2s_stride_1s"][0] == (0.0, 2.0)
    assert windows["4s_stride_2s"][0] == (0.0, 4.0)
    assert windows["8s_stride_4s"][0] == (0.0, 8.0)


def test_window_id_is_stable_for_manual_labels():
    assert window_id("206_man_plugin0_Anim_3", 4.0, 8.0) == "206_man_plugin0_Anim_3:4.000-8.000"
