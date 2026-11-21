"""Tests for like-for-like counterbalancing summaries and report failures."""

import pytest

pytest.importorskip("openpyxl")

from openpyxl import Workbook
from openpyxl import load_workbook

from te_algorithms.gdal.land_deg import (
    counterbalancing,
    counterbalancing_report,
    models,
)


def _land_type_result(code, gains, losses, achieved):
    return models.CounterbalancingLandTypeResult(
        land_type_code=code,
        land_type_name=f"Land type {code}",
        gains_area_sqkm=gains,
        losses_area_sqkm=losses,
        total_area_sqkm=10.0,
        delta_ldn=gains - losses,
        ldn_achieved=achieved,
        ldn_pct=(gains - losses) * 10,
    )


def test_neutral_only_land_type_is_included_in_results():
    summary = models.SummaryTableCounterbalancing(
        gains_by_land_type={1: 2.0},
        losses_by_land_type={1: 1.0},
        status_breakdown={1: {1: 3.0}, 2: {4: 5.0}},
    )
    assert counterbalancing._counterbalancing_land_type_codes(summary) == [1, 2]


def test_excel_total_is_labeled_as_aggregate_not_like_for_like(tmp_path):
    output = tmp_path / "counterbalancing.xlsx"
    counterbalancing_report.save_counterbalancing_excel(
        output,
        [
            _land_type_result(1, 2.0, 0.0, True),
            _land_type_result(2, 0.0, 3.0, False),
        ],
        models.SummaryTableCounterbalancing(
            gains_by_land_type={1: 2.0}, losses_by_land_type={2: 3.0}
        ),
    )
    sheet = load_workbook(output, data_only=True).active
    assert sheet["A7"].value == "Total (aggregate arithmetic balance)"
    assert sheet["E7"].value == "Aggregate balance ≥ 0: No"
    assert "not a like-for-like" in sheet["A8"].value


def test_excel_report_write_failure_is_exposed(tmp_path, monkeypatch):
    def fail_save(self, path):
        raise OSError("permission denied")

    monkeypatch.setattr(Workbook, "save", fail_save)
    with pytest.raises(RuntimeError, match="Could not save counterbalancing table"):
        counterbalancing_report.save_counterbalancing_excel(
            tmp_path / "unwritable.xlsx",
            [],
            models.SummaryTableCounterbalancing(
                gains_by_land_type={}, losses_by_land_type={}
            ),
        )


def test_ldn_planning_json_report_write_failure_is_exposed(tmp_path):
    from te_algorithms.gdal.land_deg.ldn_planning_report import save_ldn_planning_json

    with pytest.raises(RuntimeError, match="Could not save LDN Planning JSON"):
        save_ldn_planning_json(tmp_path, arr_summary={})
