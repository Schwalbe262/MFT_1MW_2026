from regression_260707.training.experimental_continuous import refresh_due


def test_refresh_due_debounces_small_dataset_growth():
    assert refresh_due(2201, 2201, 50) == (False, 2251)
    assert refresh_due(2250, 2201, 50) == (False, 2251)
    assert refresh_due(2251, 2201, 50) == (True, 2251)


def test_first_experimental_refresh_requires_two_thousand_rows():
    assert refresh_due(1999, None, 50) == (False, 2000)
    assert refresh_due(2000, None, 50) == (True, 2000)
