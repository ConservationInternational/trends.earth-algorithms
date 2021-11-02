import os
import dataclasses
import math

from typing import (
    List,
    Dict,
    Tuple,
    Optional
)
from pathlib import Path

import numpy as np
from osgeo import (
    gdal,
    osr,
)
from te_schemas import (
    schemas,
    SchemaBase,
    land_cover,
    reporting,
    aoi
)


from te_schemas.datafile import DataFile
from te_schemas.jobs import JobBand

from .. import logger
from . import util

from .util_numba import *
from .drought_numba import *

import marshmallow_dataclass


NODATA_VALUE = -32768
MASK_VALUE = -32767

POPULATION_BAND_NAME = "Population (density, persons per sq km / 10)"
SPI_BAND_NAME = "Standardized Precipitation Index (SPI)"
JRC_BAND_NAME = "Drought Vulnerability (JRC)"


@marshmallow_dataclass.dataclass
class SummaryTableDrought(SchemaBase):
    annual_area_by_drought_class: List[Dict[int, float]]
    annual_population_by_drought_class_total: List[Dict[int, float]]
    annual_population_by_drought_class_male: List[Dict[int, float]]
    annual_population_by_drought_class_female: List[Dict[int, float]]
    dvi_value_sum_and_count: Tuple[float, int]


def accumulate_drought_summary_tables(
    tables: List[SummaryTableDrought]
) -> SummaryTableDrought:
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]
        for table in tables[1:]:
            out.annual_area_by_drought_class = [
                accumulate_dicts([a,  b])
                for a, b in zip(
                    out.annual_area_by_drought_class,
                    table.annual_area_by_drought_class
                )
            ]
            out.annual_population_by_drought_class_total = [
                accumulate_dicts([a,  b])
                for a, b in zip(
                    out.annual_population_by_drought_class_total,
                    table.annual_population_by_drought_class_total
                )
            ]
            out.annual_population_by_drought_class_male = [
                accumulate_dicts([a,  b])
                for a, b in zip(
                    out.annual_population_by_drought_class_male,
                    table.annual_population_by_drought_class_male
                )
            ]
            out.annual_population_by_drought_class_female = [
                accumulate_dicts([a,  b])
                for a, b in zip(
                    out.annual_population_by_drought_class_female,
                    table.annual_population_by_drought_class_female
                )
            ]
            out.dvi_value_sum_and_count = (
                out.dvi_value_sum_and_count[0] +
                table.dvi_value_sum_and_count[0],
                out.dvi_value_sum_and_count[1] +
                table.dvi_value_sum_and_count[1]
            )
        return out


@dataclasses.dataclass()
class DroughtSummaryParams(SchemaBase):
    in_df: DataFile
    out_file: str
    drought_period: int
    mask_file: str


def _process_block(
    params: DroughtSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw
) -> Tuple[SummaryTableDrought, Dict]:

    write_arrays = {}

    # Make an array of the same size as the input arrays containing
    # the area of each cell (which is identical for all cells in a
    # given row - cell areas only vary among rows)
    cell_areas = np.repeat(
        cell_areas_raw, mask.shape[1], axis=1
    ).astype(np.float64)

    spi_rows = params.in_df.indices_for_name(SPI_BAND_NAME)
    pop_rows = params.in_df.indices_for_name(POPULATION_BAND_NAME)

    assert len(spi_rows) == len(pop_rows)

    # Calculate well annual totals of area and population exposed to drought
    annual_area_by_drought_class = []
    annual_population_by_drought_class_total = []
    annual_population_by_drought_class_male = []
    annual_population_by_drought_class_female = []
    for spi_row, pop_row in zip(spi_rows, pop_rows):
        a_drought_class = drought_class(in_array[spi_row, :, :])

        annual_area_by_drought_class.append(
            zonal_total(
                a_drought_class,
                cell_areas,
                mask
            )
        )

        a_pop = in_array[pop_row, :, :]
        a_pop_masked = a_pop.copy()
        # Account for scaling and convert from density
        a_pop_masked = a_pop * 10. * cell_areas
        a_pop_masked[a_pop == NODATA_VALUE] = 0
        annual_population_by_drought_class_total.append(
            zonal_total(
                a_drought_class,
                a_pop_masked,
                mask
            )
        )
    
    # Calculate minimum SPI in blocks of length (in years) defined by 
    # params.drought_period, and save the spi at that point as well as 
    # population that was exposed to it at that point
    for period_number, first_row in enumerate(
            range(0, len(spi_rows), params.drought_period)):
        if (first_row + params.drought_period - 1) > len(spi_rows):
            last_row = len(spi_rows)
        else:
            last_row = first_row + params.drought_period - 1

        spis = in_array[spi_rows[first_row:last_row], :, :]
        pops = in_array[pop_rows[first_row:last_row], :, :].copy()

        # Max drought is at minimum SPI
        min_indices = np.expand_dims(np.argmin(spis, axis=0), axis=0)
        max_drought = np.take_along_axis(spis, min_indices, axis=0)
        pop_at_max_drought = np.take_along_axis(pops, min_indices, axis=0)

        pop_at_max_drought_masked = pop_at_max_drought.copy()
        # Account for scaling and convert from density
        pop_at_max_drought_masked = pop_at_max_drought * 10. * cell_areas
        pop_at_max_drought_masked[pop_at_max_drought == NODATA_VALUE] = 0

        # Add one as output band numbers start at 1, not zero
        write_arrays[2*period_number + 1] = {
            'array': max_drought.squeeze(),  # remove zero dim of len 1
            'xoff': xoff,
            'yoff': yoff
        }

        # Add two as output band numbers start at 1, not zero, and this is the 
        # second band for this period
        write_arrays[2*period_number + 2] = {
            'array': pop_at_max_drought_masked.squeeze(),
            'xoff': xoff,
            'yoff': yoff
        }

    jrc_row = params.in_df.index_for_name(JRC_BAND_NAME)
    dvi_value_sum_and_count = jrc_sum_and_count(in_array[jrc_row, :, :], mask)

    return (
        SummaryTableDrought(
            annual_area_by_drought_class,
            annual_population_by_drought_class_total,
            annual_population_by_drought_class_male,
            annual_population_by_drought_class_female,
            dvi_value_sum_and_count
        ),
        write_arrays
    )


@dataclasses.dataclass()
class LineParams:
    params: DroughtSummaryParams
    image_info: util.ImageInfo
    y: int
    win_ysize: int
    lat: float



def _get_cell_areas(y, lat, win_y_size, image_info):
    cell_areas = np.array(
        [
            calc_cell_area(
                lat + image_info.pixel_height * n,
                lat + image_info.pixel_height * (n + 1),
                image_info.pixel_width
            ) for n in range(win_y_size)
        ]
    ) * 1e-6  # 1e-6 is to convert from meters to kilometers
    cell_areas.shape = (cell_areas.size, 1)

    return cell_areas

def _process_line(line_params: LineParams):

    mask_band = gdal.Open(line_params.params.mask_file).GetRasterBand(1)
    src_ds = gdal.Open(str(line_params.params.in_df.path))

    cell_areas = _get_cell_areas(
        line_params.y,
        line_params.lat,
        line_params.win_ysize,
        line_params.image_info
    )

    results = []

    for x in range(
        0,
        line_params.image_info.x_size,
        line_params.image_info.x_block_size
    ):
        if x + line_params.image_info.x_block_size < line_params.image_info.x_size:
            win_xsize = line_params.image_info.x_block_size
        else:
            win_xsize = line_params.image_info.x_size - x

        logger.debug(f'image_info: {line_params.image_info}')
        logger.debug(f'line_params: {line_params}')
        logger.debug(f'x {x}, win_xsize {win_xsize}')
        
        src_array = src_ds.ReadAsArray(
            xoff=x,
            yoff=line_params.y,
            xsize=win_xsize,
            ysize=line_params.win_ysize
        )

        mask_array = mask_band.ReadAsArray(
            xoff=x,
            yoff=line_params.y,
            win_xsize=win_xsize,
            win_ysize=line_params.win_ysize
        )
        mask_array = mask_array == MASK_VALUE

        result = _process_block(
            line_params.params,
            src_array,
            mask_array,
            x,
            line_params.y,
            cell_areas
        )

        results.append(result)

    return results


@dataclasses.dataclass()
class DroughtSummary:
    params: DroughtSummaryParams

    def __init__(self):
        self.image_info = util.get_image_info(self.params.in_df.path)

    def is_killed(self):
        return False

    def emit_progress(self, *args, **kwargs):
        pass

    def get_line_params(self):
        '''Make a list of parameters to use in the _process_line function'''
        # Set initial lat to the top left corner latitude
        self.src_ds = gdal.Open(str(self.params.in_df.path))
        src_gt = self.src_ds.GetGeoTransform()
        lat = src_gt[3]

        line_params = []
        for y in range(0, self.image_info.y_size, self.image_info.y_block_size):
            if y + self.image_info.y_block_size < self.image_info.y_size:
                win_ysize = self.image_info.y_block_size
            else:
                win_ysize = self.image_info.y_size - y

            line_params.append(LineParams(
                self.params, self.image_info, y, win_ysize, lat))

            lat += self.image_info.pixel_height * win_ysize

        return line_params

    def process_lines(self, line_params_list):
        out_ds = self._get_out_ds()

        out = []
        for n, line_params in enumerate(line_params_list):
            self.emit_progress(n / len(line_params_list))
            results = _process_line(
                line_params
            )

            for result in results:
                out.append(result[0])
                for key, value in result[1].items():
                    logger.debug(f'key {key}, value {value}')
                    out_ds.GetRasterBand(key).WriteArray(**value)

        out = accumulate_drought_summary_tables(out)

        return out

    # def multiprocess_lines(self, line_params_list):
    #     out_ds = self._get_out_ds()
    #
    #     out = []
    #     with multiprocessing.Pool(3) as p:
    #         for result in p.imap_unordered(
    #             _process_line,
    #             line_params_list,
    #             chunksize=20
    #         ):
    #             if self.is_killed():
    #                 p.terminate()
    #                 break
    #             # self.emit_progress(n / len(line_params_list))
    #
    #             out.append(result[0])
    #             for key, value in result[1].items():
    #                 logger.debug(f'key {key}, value {value}')
    #                 out_ds.GetRasterBand(key).WriteArray(**value)
    #
    #     out = accumulate_drought_summary_tables(out)
    #
    #     return out

    def _get_out_ds(self):
        n_out_bands = int(
            2 * math.ceil(
                len(
                    self.params.in_df.indices_for_name(SPI_BAND_NAME)
                ) / self.params.drought_period
            ) + 1
        )
        out_ds = util.setup_output_image(
            self.params.in_df.path,
            self.params.out_file,
            n_out_bands,
            self.image_info
        )

        return out_ds


def save_summary_table_excel(
        output_path: Path,
        summary_table: SummaryTableDrought,
        years: List[int]
):
    """Save summary table into an xlsx file on disk"""
    template_summary_table_path = Path(
        __file__).parents[1] / "data/summary_table_drought.xlsx"
    workbook = openpyxl.load_workbook(str(template_summary_table_path))
    _render_drought_workbook(
        workbook,
        summary_table,
        years
    )
    try:
        workbook.save(output_path)
        log(u'Indicator table saved to {}'.format(output_path))

    except IOError:
        error_message = (
            f"Error saving output table - check that {output_path!r} is accessible "
            f"and not already open."
        )
        log(error_message)


def _get_population_list_by_drought_class(pop_by_drought, pop_type):
    return reporting.PopulationList(
        'Total population by drought class',
        [
            reporting.Population(
                'Mild drought',
                pop_by_drought.get(1, 0.),
                type=pop_type
            ),
            reporting.Population(
                'Moderate drought',
                pop_by_drought.get(2, 0.),
                type=pop_type
            ),
            reporting.Population(
                'Severe drought',
                pop_by_drought.get(3, 0.),
                type=pop_type
            ),
            reporting.Population(
                'Extreme drought',
                pop_by_drought.get(4, 0.),
                type=pop_type
            ),
            reporting.Population(
                'Non-drought',
                pop_by_drought.get(0, 0.),
                type=pop_type
            ),
            reporting.Population(
                'No data',
                pop_by_drought.get(-32768, 0.),
                type=pop_type
            )
        ]
    )

def save_reporting_json(
    output_path: Path,
    st: SummaryTableDrought,
    params: dict,
    task_name: str,
    aoi: aoi.AOI,
    summary_table_kwargs: dict
):

    drought_tier_one = {}
    drought_tier_two = {}

    for n, year in enumerate(range(
        int(params['layer_spi_years'][0]),
        int(params['layer_spi_years'][-1]) + 1
    )):

        drought_tier_one[year] = reporting.AreaList(
            "Area by drought class",
            'sq km',
            [
                reporting.Area(
                    'Mild drought',
                    st.annual_area_by_drought_class[n].get(1, 0.)),
                reporting.Area(
                    'Moderate drought',
                    st.annual_area_by_drought_class[n].get(2, 0.)),
                reporting.Area(
                    'Severe drought',
                    st.annual_area_by_drought_class[n].get(3, 0.)),
                reporting.Area(
                    'Extreme drought',
                    st.annual_area_by_drought_class[n].get(4, 0.)),
                reporting.Area(
                    'Non-drought',
                    st.annual_area_by_drought_class[n].get(0, 0.)),
                reporting.Area(
                    'No data',
                    st.annual_area_by_drought_class[n].get(-32768, 0.))
            ]
        )

        drought_tier_two[year] = {
            'Total population': _get_population_list_by_drought_class(
                st.annual_population_by_drought_class_total[n],
                "Total population"
            )
        }
        if st.annual_population_by_drought_class_male:
            drought_tier_two[year]['Male population'] = _get_population_list_by_drought_class(
                _get_population_list_by_drought_class(
                    st.annual_population_by_drought_class_male[n],
                    "Male population"
                )
            )
        if st.annual_population_by_drought_class_female:
            drought_tier_two[year]['Female population'] = _get_population_list_by_drought_class(
                _get_population_list_by_drought_class(
                    st.annual_population_by_drought_class_female[n],
                    "Female population"
                )
            )

    drought_tier_three = {
        2018: reporting.Value(
            'Mean value',
            st.dvi_value_sum_and_count[0] / st.dvi_value_sum_and_count[1]
        )
    }

    ##########################################################################
    # Format final JSON output
    te_summary = reporting.TrendsEarthDroughtSummary(
            metadata=reporting.ReportMetadata(
                title='Trends.Earth Summary Report',
                date=dt.datetime.now(dt.timezone.utc),

                trends_earth_version=schemas.TrendsEarthVersion(
                    version=__version__,
                    revision=__revision__,
                    release_date=dt.datetime.strptime(
                        __release_date__,
                        '%Y/%m/%d %H:%M:%SZ'
                    )
                ),

                area_of_interest=schemas.AreaOfInterest(
                    name=task_name,  # TODO replace this with area of interest name once implemented in TE
                    geojson=aoi.get_geojson(),
                    crs_wkt=aoi.get_crs_wkt()
                )
            ),

            drought=reporting.DroughtReport(
                tier_one=drought_tier_one,
                tier_two=drought_tier_two,
                tier_three=drought_tier_three
            )
        )

    try:
        te_summary_json = json.loads(
            reporting.TrendsEarthDroughtSummary.Schema().dumps(te_summary)
        )
        with open(output_path, 'w') as f:
            json.dump(te_summary_json, f, indent=4)

        return True

    except IOError:
        log(u'Error saving {}'.format(output_path))
        error_message = (
            "Error saving indicator table JSON - check that "
            f"{output_path} is accessible and not already open."
        )

        return False


def _render_drought_workbook(
    template_workbook,
    summary_table: SummaryTableDrought,
    years: List[int]
):
    _write_drought_area_sheet(
        template_workbook["Area under drought by year"],
        summary_table,
        years
    )

    _write_drought_pop_total_sheet(
        template_workbook["Pop under drought (total)"],
        summary_table,
        years
    )

    _write_dvi_sheet(
        template_workbook["Drought Vulnerability Index"],
        summary_table,
        years
    )

    return template_workbook


def _get_col_for_drought_class(
    annual_values_by_drought,
    drought_code
):
    out = []
    for values_by_drought in annual_values_by_drought:
        out.append(values_by_drought.get(drought_code, 0.))
    return np.array(out)


def _write_dvi_sheet(
    sheet,
    st: SummaryTableDrought,
    years
):

    # Make this more informative when fuller DVI calculations are available...
    cell = sheet.cell(6, 2)
    cell.value = 2018
    cell = sheet.cell(6, 3)
    cell.value = st.dvi_value_sum_and_count[0] / st.dvi_value_sum_and_count[1]

    utils.maybe_add_image_to_sheet(
        "trends_earth_logo_bl_300width.png", sheet, "H1")


def _write_drought_area_sheet(
    sheet,
    st: SummaryTableDrought,
    years
):
    summary.write_col_to_sheet(
        sheet,
        np.array(years),
        2, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, 1),
        4, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, 2),
        6, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, 3),
        8, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, 4),
        10, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, 0),
        12, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_area_by_drought_class, -32768),
        14, 7
    )
    utils.maybe_add_image_to_sheet(
        "trends_earth_logo_bl_300width.png", sheet, "L1")


def _write_drought_pop_total_sheet(
    sheet,
    st: SummaryTableDrought,
    years
):
    summary.write_col_to_sheet(
        sheet,
        np.array(years),
        2, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, 1),
        4, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, 2),
        6, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, 3),
        8, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, 4),
        10, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, 0),
        12, 7
    )
    summary.write_col_to_sheet(
        sheet,
        _get_col_for_drought_class(
            st.annual_population_by_drought_class_total, -32768),
        14, 7
    )
    utils.maybe_add_image_to_sheet(
        "trends_earth_logo_bl_300width.png", sheet, "L1")

