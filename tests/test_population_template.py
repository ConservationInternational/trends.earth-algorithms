from pathlib import Path

from openpyxl import load_workbook


def test_population_summary_formulas_match_sdg_area_categories():
    workbook_path = (
        Path(__file__).parents[1]
        / "te_algorithms"
        / "data"
        / "summary_table_ld_sdg.xlsx"
    )
    workbook = load_workbook(workbook_path, data_only=False)
    population_sheet = workbook["Population"]

    assert [population_sheet[f"C{row}"].value for row in range(6, 10)] == [
        "Improved land:",
        "Stable land:",
        "Degraded land:",
        "No data:",
    ]
    assert [population_sheet[f"D{row}"].value for row in range(6, 10)] == [
        "='SDG 15.3.1'!F6",
        "='SDG 15.3.1'!F7",
        "='SDG 15.3.1'!F8",
        "='SDG 15.3.1'!F9",
    ]
