from keo.core.schema import EXPECTED_EXPERIMENT_COLUMNS


def test_schema_has_expected_columns():
    assert "experiment_id" in EXPECTED_EXPERIMENT_COLUMNS
    assert "cv_metric" in EXPECTED_EXPERIMENT_COLUMNS
