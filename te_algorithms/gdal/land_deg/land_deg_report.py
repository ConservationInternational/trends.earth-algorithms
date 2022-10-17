import datetime as dt
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import openpyxl
from te_schemas import land_cover
from te_schemas import reporting
from te_schemas import schemas
from te_schemas.aoi import AOI

from . import config
from . import models
from .. import xl
from ... import __release_date__
from ... import __version__

logger = logging.getLogger(__name__)


def save_summary_table_excel(
    output_path: Path,
    summary_table: models.SummaryTableLD,
    periods,
    land_cover_years: List[int],
    soil_organic_carbon_years: List[int],
    lc_legend_nesting: land_cover.LCLegendNesting,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
    period_name,
):
    """Save summary table into an xlsx file on disk"""
    template_summary_table_path = (
        Path(__file__).parents[2] / "data/summary_table_ld_sdg.xlsx"
    )
    workbook = openpyxl.load_workbook(str(template_summary_table_path))
    _render_ld_workbook(
        workbook,
        summary_table,
        periods,
        land_cover_years,
        soil_organic_carbon_years,
        lc_legend_nesting,
        lc_trans_matrix,
        period_name,
    )
    try:
        workbook.save(output_path)
        logger.info("Indicator table saved to {}".format(output_path))

    except IOError:
        error_message = (
            f"Error saving output table - check that {output_path!r} is accessible "
            f"and not already open."
        )
        logger.error(error_message)


def _get_population_list_by_degradation_class(pop_by_deg_class, pop_type):
    return reporting.PopulationList(
        "Population by degradation class",
        [
            reporting.Population("Improved", pop_by_deg_class.get(1, 0), type=pop_type),
            reporting.Population("Stable", pop_by_deg_class.get(0, 0), type=pop_type),
            reporting.Population(
                "Degraded", pop_by_deg_class.get(-1, 0), type=pop_type
            ),
            reporting.Population(
                "No data", pop_by_deg_class.get(config.NODATA_VALUE, 0), type=pop_type
            ),
        ],
    )


def save_reporting_json(
    output_path: Path,
    summary_tables: List[models.SummaryTableLD],
    summary_table_progress: models.SummaryTableLDProgress,
    params: dict,
    task_name: str,
    aoi: AOI,
    summary_table_kwargs: dict,
):

    land_condition_reports = {}
    affected_pop_reports = {}

    for period_name, period_params in params.items():
        st = summary_tables[period_name]

        ##########################################################################
        # Area summary tables
        lc_legend_nesting = summary_table_kwargs[period_name]["lc_legend_nesting"]
        lc_trans_matrix = summary_table_kwargs[period_name]["lc_trans_matrix"]

        sdg_summary = reporting.AreaList(
            "SDG Indicator 15.3.1",
            "sq km",
            [
                reporting.Area("Improved", st.sdg_summary.get(1, 0.0)),
                reporting.Area("Stable", st.sdg_summary.get(0, 0.0)),
                reporting.Area("Degraded", st.sdg_summary.get(-1, 0.0)),
                reporting.Area("No data", st.sdg_summary.get(config.NODATA_VALUE, 0)),
            ],
        )

        prod_summaries = {
            key: reporting.AreaList(
                "Productivity",
                "sq km",
                [
                    reporting.Area("Improved", value.get(1, 0.0)),
                    reporting.Area("Stable", value.get(0, 0.0)),
                    reporting.Area("Degraded", value.get(-1, 0.0)),
                    reporting.Area("No data", value.get(config.NODATA_VALUE, 0)),
                ],
            )
            for key, value in st.prod_summary.items()
        }

        soc_summaries = {
            key: reporting.AreaList(
                "Soil organic carbon",
                "sq km",
                [
                    reporting.Area("Improved", value.get(1, 0.0)),
                    reporting.Area("Stable", value.get(0, 0.0)),
                    reporting.Area("Degraded", value.get(-1, 0.0)),
                    reporting.Area("No data", value.get(config.NODATA_VALUE, 0.0)),
                ],
            )
            for key, value in st.soc_summary.items()
        }

        lc_summary = reporting.AreaList(
            "Land cover",
            "sq km",
            [
                reporting.Area("Improved", st.lc_summary.get(1, 0.0)),
                reporting.Area("Stable", st.lc_summary.get(0, 0.0)),
                reporting.Area("Degraded", st.lc_summary.get(-1, 0.0)),
                reporting.Area("No data", st.lc_summary.get(config.NODATA_VALUE, 0.0)),
            ],
        )

        #######################################################################
        # Productivity tables
        lc_trans_dict = lc_trans_matrix.get_transition_integers_key()

        crosstab_prod = []
        # If no land cover data was available for first year of productivity
        # data, then won't be able to output these tables, so need to check
        # first if data is available

        if len([*st.lc_trans_prod_bizonal.keys()]) > 0:
            for prod_name, prod_code in config.PRODUCTIVITY_CLASS_KEY.items():
                crosstab_entries = []

                for transition, codes in lc_trans_dict.items():
                    initial_class = lc_trans_matrix.legend.classByCode(
                        codes["initial"]
                    ).get_name()
                    final_class = lc_trans_matrix.legend.classByCode(
                        codes["final"]
                    ).get_name()
                    crosstab_entries.append(
                        reporting.CrossTabEntry(
                            initial_class,
                            final_class,
                            value=st.lc_trans_prod_bizonal.get(
                                (transition, prod_code), 0.0
                            ),
                        )
                    )
                crosstab_prod.append(
                    reporting.CrossTab(
                        prod_name,
                        unit="sq km",
                        initial_year=period_params["periods"]["productivity"][
                            "year_initial"
                        ],
                        final_year=period_params["periods"]["productivity"][
                            "year_final"
                        ],
                        values=crosstab_entries,
                    )
                )

        #######################################################################
        # Land cover tables
        land_cover_years = period_params["layer_lc_years"]

        ###
        # LC transition cross tabs
        crosstab_lcs = []

        for lc_trans_zonal_areas, lc_trans_zonal_areas_period in zip(
            st.lc_trans_zonal_areas, st.lc_trans_zonal_areas_periods
        ):
            lc_by_transition_type = []

            for transition, codes in lc_trans_dict.items():
                initial_class = lc_trans_matrix.legend.classByCode(
                    codes["initial"]
                ).get_name()
                final_class = lc_trans_matrix.legend.classByCode(
                    codes["final"]
                ).get_name()
                lc_by_transition_type.append(
                    reporting.CrossTabEntry(
                        initial_class,
                        final_class,
                        value=lc_trans_zonal_areas.get(transition, 0.0),
                    )
                )
            lc_by_transition_type = sorted(
                lc_by_transition_type, key=lambda i: i.value, reverse=True
            )
            crosstab_lc = reporting.CrossTab(
                name="Land area by land cover transition type",
                unit="sq km",
                initial_year=lc_trans_zonal_areas_period["year_initial"],
                final_year=lc_trans_zonal_areas_period["year_final"],
                # TODO: Check indexing as may be missing a class
                values=lc_by_transition_type,
            )
            crosstab_lcs.append(crosstab_lc)

        ###
        # LC by year
        lc_by_year = {}

        for year_num, year in enumerate(land_cover_years):
            total_land_area = sum(
                [
                    value
                    for key, value in st.lc_annual_totals[year_num].items()
                    if key != config.MASK_VALUE
                ]
            )
            logging.debug(
                f"Total land area in {year} per land cover data {total_land_area}"
            )

            for lc_class in lc_trans_matrix.legend.key:
                logging.debug(
                    f"Total area of {lc_class.get_name()} in {year}: {st.lc_annual_totals[year_num].get(lc_class.code, 0.)}"
                )

            lc_by_year[int(year)] = {
                lc_class.get_name(): st.lc_annual_totals[year_num].get(
                    lc_class.code, 0.0
                )
                for lc_class in lc_trans_matrix.legend.key
                + [lc_trans_matrix.legend.nodata]
            }
        lc_by_year_by_class = reporting.ValuesByYearDict(
            name="Area by year by land cover class", unit="sq km", values=lc_by_year
        )

        #######################################################################
        # Soil organic carbon tables
        soil_organic_carbon_years = period_params["layer_soc_years"]

        ###
        # SOC by transition type (initial and final stock for each transition
        # type)
        soc_by_transition = []
        # Note that the last element is skipped, as it is water, and don't want
        # to count water in SOC totals

        # TODO: Right now the below is produced even if the initial and final years of
        # the SOC data don't match those of the LC data. Does this make sense? That
        # would mean tabulating SOC change for years that could potentially be very
        # different from the LC years (if both use custom data)

        for transition, codes in lc_trans_dict.items():
            initial_class = lc_trans_matrix.legend.classByCode(
                codes["initial"]
            ).get_name()
            final_class = lc_trans_matrix.legend.classByCode(codes["final"]).get_name()
            soc_by_transition.append(
                reporting.CrossTabEntryInitialFinal(
                    initial_label=initial_class,
                    final_label=final_class,
                    initial_value=st.lc_trans_zonal_soc_initial.get(transition, 0.0),
                    final_value=st.lc_trans_zonal_soc_final.get(transition, 0.0),
                )
            )
        initial_soc_year = period_params["layer_soc_deg_years"]["year_initial"]
        final_soc_year = period_params["layer_soc_deg_years"]["year_final"]
        crosstab_soc_by_transition_per_ha = reporting.CrossTab(
            name="Initial and final carbon stock by transition type",
            unit="tons",
            initial_year=initial_soc_year,
            final_year=final_soc_year,
            values=soc_by_transition,
        )

        ###
        # SOC by year by land cover class
        soc_by_year = {}

        for year_num, soc_by_lc_annual_total in enumerate(st.soc_by_lc_annual_totals):
            year = soil_organic_carbon_years[year_num]
            soc_by_year[int(year)] = {
                lc_class.get_name(): soc_by_lc_annual_total.get(lc_class.code, 0.0)
                for lc_class in lc_trans_matrix.legend.key
                + [lc_trans_matrix.legend.nodata]
            }

        soc_by_year_by_class = reporting.ValuesByYearDict(
            name="Soil organic carbon by year by land cover class",
            unit="tonnes",
            values=soc_by_year,
        )

        ###
        # Setup this period's land condition report
        land_condition_reports[period_name] = reporting.LandConditionReport(
            sdg=reporting.SDG15Report(summary=sdg_summary),
            productivity=reporting.ProductivityReport(
                summaries=prod_summaries, crosstabs_by_productivity_class=crosstab_prod
            ),
            land_cover=reporting.LandCoverReport(
                summary=lc_summary,
                legend_nesting=lc_legend_nesting,
                transition_matrix=lc_trans_matrix,
                crosstabs_by_land_cover_class=crosstab_lcs,
                land_cover_areas_by_year=lc_by_year_by_class,
            ),
            soil_organic_carbon=reporting.SoilOrganicCarbonReport(
                summaries=soc_summaries,
                crosstab_by_land_cover_class=crosstab_soc_by_transition_per_ha,
                soc_stock_by_year=soc_by_year_by_class,
            ),
        )

        ###
        # Setup this period's affected population report

        affected_by_deg_summary = {
            "Total population": _get_population_list_by_degradation_class(
                st.sdg_zonal_population_total, "Total population"
            )
        }

        if len(st.sdg_zonal_population_female) > 0:
            affected_by_deg_summary[
                "Female population"
            ] = _get_population_list_by_degradation_class(
                st.sdg_zonal_population_female, "Female population"
            )

        if len(st.sdg_zonal_population_male) > 0:
            affected_by_deg_summary[
                "Male population"
            ] = _get_population_list_by_degradation_class(
                st.sdg_zonal_population_male, "Male population"
            )

        affected_pop_reports[period_name] = reporting.AffectedPopulationReport(
            affected_by_deg_summary
        )

    if summary_table_progress:
        land_condition_reports["integrated"] = reporting.LandConditionProgressReport(
            sdg=reporting.AreaList(
                "SDG Indicator 15.3.1 (progress since baseline)",
                "sq km",
                [
                    reporting.Area(
                        "Improved", summary_table_progress.sdg_summary.get(1, 0.0)
                    ),
                    reporting.Area(
                        "Stable", summary_table_progress.sdg_summary.get(0, 0.0)
                    ),
                    reporting.Area(
                        "Degraded", summary_table_progress.sdg_summary.get(-1, 0.0)
                    ),
                    reporting.Area(
                        "No data",
                        summary_table_progress.sdg_summary.get(config.NODATA_VALUE, 0),
                    ),
                ],
            ),
            productivity={
                key: reporting.AreaList(
                    "Productivity (progress since baseline)",
                    "sq km",
                    [
                        reporting.Area("Improved", value.get(1, 0.0)),
                        reporting.Area("Stable", value.get(0, 0.0)),
                        reporting.Area("Degraded", value.get(-1, 0.0)),
                        reporting.Area("No data", value.get(config.NODATA_VALUE, 0)),
                    ],
                )
                for key, value in summary_table_progress.prod_summary.items()
            },
            land_cover=reporting.AreaList(
                "Land cover (progress since baseline)",
                "sq km",
                [
                    reporting.Area(
                        "Improved", summary_table_progress.lc_summary.get(1, 0.0)
                    ),
                    reporting.Area(
                        "Stable", summary_table_progress.lc_summary.get(0, 0.0)
                    ),
                    reporting.Area(
                        "Degraded", summary_table_progress.lc_summary.get(-1, 0.0)
                    ),
                    reporting.Area(
                        "No data",
                        summary_table_progress.lc_summary.get(config.NODATA_VALUE, 0),
                    ),
                ],
            ),
            soil_organic_carbon={
                key: reporting.AreaList(
                    "Soil organic carbon (progress since baseline)",
                    "sq km",
                    [
                        reporting.Area("Improved", value.get(1, 0.0)),
                        reporting.Area("Stable", value.get(0, 0.0)),
                        reporting.Area("Degraded", value.get(-1, 0.0)),
                        reporting.Area("No data", value.get(config.NODATA_VALUE, 0)),
                    ],
                )
                for key, value in summary_table_progress.prod_summary.items()
            },
        )

    ##########################################################################
    # Format final JSON output
    te_summary = reporting.TrendsEarthLandConditionSummary(
        metadata=reporting.ReportMetadata(
            title="Trends.Earth Summary Report",
            date=dt.datetime.now(dt.timezone.utc),
            trends_earth_version=schemas.TrendsEarthVersion(
                version=__version__,
                revision=None,
                release_date=dt.datetime.strptime(
                    __release_date__, "%Y/%m/%d %H:%M:%SZ"
                ),
            ),
            area_of_interest=schemas.AreaOfInterest(
                name=task_name,  # TODO replace this with area of interest name once implemented in TE
                geojson=aoi.get_geojson(),
                crs_wkt=aoi.get_crs_wkt(),
            ),
        ),
        land_condition=land_condition_reports,
        affected_population=affected_pop_reports,
    )

    try:
        te_summary_json = json.loads(
            reporting.TrendsEarthLandConditionSummary.Schema().dumps(te_summary)
        )
        with open(output_path, "w") as f:
            json.dump(te_summary_json, f, indent=4)

        return te_summary_json

    except IOError:
        logger.error(
            "Error saving indicator table JSON - check that "
            f"{output_path} is accessible and not already open."
        )

        return None


def _render_ld_workbook(
    template_workbook,
    summary_table: models.SummaryTableLD,
    periods,
    lc_years: List[int],
    soc_years: List[int],
    lc_legend_nesting: land_cover.LCLegendNesting,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
    period_name,
):
    _write_overview_sheet(template_workbook["SDG 15.3.1"], summary_table)
    _write_productivity_sheet(
        template_workbook["Productivity"], summary_table, lc_trans_matrix
    )
    _write_soc_sheet(
        template_workbook["Soil organic carbon"],
        summary_table,
        lc_trans_matrix,
    )
    _write_land_cover_sheet(
        template_workbook["Land cover"],
        summary_table,
        lc_trans_matrix,
        periods["land_cover"],
    )
    _write_population_sheet(template_workbook["Population"], summary_table)

    return template_workbook


def _get_summary_array(d):
    """pulls summary values for excel sheet from a summary dictionary"""

    return np.array([d.get(1, 0), d.get(0, 0), d.get(-1, 0), d.get(-32768, 0)])


def _write_overview_sheet(sheet, summary_table: models.SummaryTableLD):
    xl.write_col_to_sheet(sheet, _get_summary_array(summary_table.sdg_summary), 6, 6)
    xl.maybe_add_image_to_sheet("trends_earth_logo_bl_300width.png", sheet)


def _write_productivity_sheet(
    sheet,
    st: models.SummaryTableLD,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
):
    xl.write_col_to_sheet(
        sheet, _get_summary_array(st.prod_summary["all_cover_types"]), 6, 6
    )

    if len([*st.lc_trans_prod_bizonal.keys()]) > 0:
        # If no land cover data was available for first year of productivity
        # data, then won't be able to output these tables
        xl.write_table_to_sheet(
            sheet, _get_prod_table(st.lc_trans_prod_bizonal, 5, lc_trans_matrix), 16, 3
        )
        xl.write_table_to_sheet(
            sheet, _get_prod_table(st.lc_trans_prod_bizonal, 4, lc_trans_matrix), 28, 3
        )
        xl.write_table_to_sheet(
            sheet, _get_prod_table(st.lc_trans_prod_bizonal, 3, lc_trans_matrix), 40, 3
        )
        xl.write_table_to_sheet(
            sheet, _get_prod_table(st.lc_trans_prod_bizonal, 2, lc_trans_matrix), 52, 3
        )
        xl.write_table_to_sheet(
            sheet, _get_prod_table(st.lc_trans_prod_bizonal, 1, lc_trans_matrix), 64, 3
        )
        xl.write_table_to_sheet(
            sheet,
            _get_prod_table(
                st.lc_trans_prod_bizonal, config.NODATA_VALUE, lc_trans_matrix
            ),
            76,
            3,
        )
    xl.maybe_add_image_to_sheet("trends_earth_logo_bl_300width.png", sheet)


def _write_soc_sheet(
    sheet,
    st: models.SummaryTableLD,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
):
    xl.write_col_to_sheet(
        sheet, _get_summary_array(st.soc_summary["all_cover_types"]), 6, 6
    )

    # First write baseline

    if st.soc_by_lc_annual_totals != []:
        xl.write_col_to_sheet(
            sheet,
            _get_totals_by_lc_class_as_array(
                st.soc_by_lc_annual_totals[0],
                lc_trans_matrix,
                excluded_codes=[6],  # exclude water
            ),
            7,
            16,
        )
        # Now write target
        xl.write_col_to_sheet(
            sheet,
            _get_totals_by_lc_class_as_array(
                st.soc_by_lc_annual_totals[-1],
                lc_trans_matrix,
                excluded_codes=[6],  # exclude water
            ),
            8,
            16,
        )

    if st.lc_annual_totals != []:
        # Write table of baseline areas
        lc_bl_no_water = _get_totals_by_lc_class_as_array(
            st.lc_annual_totals[0], lc_trans_matrix, excluded_codes=[6]  # exclude water
        )
        xl.write_col_to_sheet(sheet, lc_bl_no_water, 5, 16)
        # Write table of final year areas
        lc_final_no_water = _get_totals_by_lc_class_as_array(
            st.lc_annual_totals[-1],
            lc_trans_matrix,
            excluded_codes=[6],  # exclude water
        )
        xl.write_col_to_sheet(sheet, lc_final_no_water, 6, 16)

    if st.lc_trans_zonal_soc_initial != {} and st.lc_trans_zonal_soc_final != {}:
        # write_soc_stock_change_table has its own writing function as it needs
        # to write a
        # mix of numbers and strings
        _write_soc_stock_change_table(
            sheet,
            27,
            3,
            st.lc_trans_zonal_soc_initial,
            st.lc_trans_zonal_soc_final,
            lc_trans_matrix,
            excluded_codes=[6],  # exclude water
        )
    xl.maybe_add_image_to_sheet("trends_earth_logo_bl_300width.png", sheet)


def _write_land_cover_sheet(
    sheet,
    st: models.SummaryTableLD,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
    period,
):
    lc_trans_zonal_areas = [
        x
        for x, p in zip(st.lc_trans_zonal_areas, st.lc_trans_zonal_areas_periods)
        if p == period
    ][0]

    xl.write_col_to_sheet(sheet, _get_summary_array(st.lc_summary), 6, 6)
    xl.write_table_to_sheet(
        sheet, _get_lc_trans_table(lc_trans_zonal_areas, lc_trans_matrix), 26, 3
    )
    xl.maybe_add_image_to_sheet("trends_earth_logo_bl_300width.png", sheet)


def _write_population_sheet(sheet, st: models.SummaryTableLD):

    xl.write_col_to_sheet(
        sheet, _get_summary_array(st.sdg_zonal_population_total), 6, 6
    )
    xl.maybe_add_image_to_sheet("trends_earth_logo_bl_300width.png", sheet)


def _get_prod_table(lc_trans_prod_bizonal, prod_code, lc_trans_matrix):
    lc_codes = sorted([c.code for c in lc_trans_matrix.legend.key])
    out = np.zeros((len(lc_codes), len(lc_codes)))

    for i, i_code in enumerate(lc_codes):
        for f, f_code in enumerate(lc_codes):
            transition = i_code * lc_trans_matrix.legend.get_multiplier() + f_code
            out[i, f] = lc_trans_prod_bizonal.get((transition, prod_code), 0.0)

    return out


def _get_totals_by_lc_class_as_array(
    annual_totals,
    lc_trans_matrix,
    excluded_codes=[],  # to exclude water when used on SOC table
):
    lc_codes = sorted(
        [c.code for c in lc_trans_matrix.legend.key if c.code not in excluded_codes]
    )

    return np.array([annual_totals.get(lc_code, 0.0) for lc_code in lc_codes])


def _write_soc_stock_change_table(
    sheet,
    first_row,
    first_col,
    soc_bl_totals,
    soc_final_totals,
    lc_trans_matrix,
    excluded_codes=[],  # to exclude water
):
    lc_codes = sorted(
        [c.code for c in lc_trans_matrix.legend.key if c.code not in excluded_codes]
    )

    for i, i_code in enumerate(lc_codes):
        for f, f_code in enumerate(lc_codes):
            cell = sheet.cell(row=i + first_row, column=f + first_col)
            transition = i_code * lc_trans_matrix.legend.get_multiplier() + f_code
            bl_soc = soc_bl_totals.get(transition, 0.0)
            final_soc = soc_final_totals.get(transition, 0.0)
            try:
                cell.value = (final_soc - bl_soc) / bl_soc
            except ZeroDivisionError:
                cell.value = ""


def _get_lc_trans_table(lc_trans_totals, lc_trans_matrix, excluded_codes=[]):
    lc_codes = sorted(
        [c.code for c in lc_trans_matrix.legend.key if c.code not in excluded_codes]
    )
    out = np.zeros((len(lc_codes), len(lc_codes)))

    for i, i_code in enumerate(lc_codes):
        for f, f_code in enumerate(lc_codes):
            transition = i_code * lc_trans_matrix.legend.get_multiplier() + f_code
            out[i, f] = lc_trans_totals.get(transition, 0.0)

    return out


def _get_soc_total(soc_table, transition):
    ind = np.where(soc_table[0] == transition)[0]

    if ind.size == 0:
        return 0
    else:
        return float(soc_table[1][ind])
