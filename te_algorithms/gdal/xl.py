from pathlib import Path

from numpy import ndarray


def maybe_add_image_to_sheet(image_filename: str, sheet, place="H1"):
    try:
        from openpyxl.drawing.image import Image

        image_path = Path(__file__).parents[1] / "data" / image_filename
        logo = Image(image_path)
        sheet.add_image(logo, place)
    except ImportError:
        # add_image will fail on computers without PIL installed (this will be
        # an issue on some Macs, likely others). it is only used here to add
        # our logo, so no big deal.
        pass


def write_col_to_sheet(sheet, d, col, first_row):
    if isinstance(d, ndarray):
        length = d.size
    else:
        length = len(d)
    for row in range(length):
        cell = sheet.cell(row=row + first_row, column=col)
        cell.value = d[row]


def write_row_to_sheet(sheet, d, row, first_col):
    if isinstance(d, ndarray):
        length = d.size
    else:
        length = len(d)
    for col in range(length):
        cell = sheet.cell(row=row, column=first_col + col)
        cell.value = d[col]


def write_table_to_sheet(sheet, d, first_row, first_col):
    for row in range(d.shape[0]):
        for col in range(d.shape[1]):
            cell = sheet.cell(row=row + first_row, column=col + first_col)
            cell.value = d[row, col]
