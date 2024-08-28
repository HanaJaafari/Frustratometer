import pytest
from frustratometer.utils.format_time import format_time  # Assume the format_time function is in time_formatter.py

@pytest.mark.parametrize("seconds, expected_output", [
    (1e-25, "0.1 yoctoseconds (0.1 ys)"),
    (1e-22, "100 yoctoseconds (100 ys)"),
    (1e-19, "100 zeptoseconds (100 zs)"),
    (1e-16, "100 attoseconds (100 as)"),
    (0.000000001, "1 nanosecond (1 ns)"),
    (0.000001, "1 microsecond (1 μs)"),
    (0.001, "1 millisecond (1 ms)"),
    (1, "1 second (1 s)"),
    (30, "30 seconds (30 s)"),
    (60, "1 minute (60 s)"),
    (3600, "1 hour (3.6 ks)"),
    (3900, "1.08 hours (3.9 ks)"),
    (86400, "1 day (86.4 ks)"),
    (604800, "1 week (604.8 ks)"),
    (2630016, "1 month (2.63 Ms)"),
    (31557600, "1 year (31.56 Ms)"),
    (315576000, "1 decade (315.58 Ms)"),
    (3155760000, "1 century (3.16 Gs)"),
    (2*3155760000, "2 centuries (6.31 Gs)"),
    (31557600000, "1 thousand years (31.56 Gs)"),
    (3.15576e13, "1 million years (31.56 Ts)"),
    (3.15576e16, "1 billion years (31.56 Ps)"),
    (-3600, "-1 hour (-3.6 ks)"),
    (0, "0 seconds (0 s)"),
])
def test_format_time(seconds, expected_output):
    assert format_time(seconds) == expected_output

# Additional tests for edge cases
def test_very_large_time():
    assert format_time(1e25) == "317.1 billion years (10 Ys)"

def test_very_small_time():
    assert format_time(1e-30) == "0.01 yoctoseconds (0.01 ys)"

# Test for floating point precision
@pytest.mark.parametrize("seconds, expected_output", [
    (1.23456789, "1.23 seconds (1.23 s)"),
    (123456.789, "1 day 10 hours 17 minutes 36.79 seconds (123.46 ks)"),
])
def test_floating_point_precision(seconds, expected_output):
    assert format_time(seconds) == expected_output

# Test for negative times
@pytest.mark.parametrize("seconds, expected_output", [
    (-60, "1 minute ago (-60 s)"),
    (-86400, "1 day ago (-86.4 ks)"),
    (-31536000, "1 year ago (-31.54 Ms)"),
])
def test_negative_times(seconds, expected_output):
    assert format_time(seconds) == expected_output