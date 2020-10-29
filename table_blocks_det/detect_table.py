"""Detection of table and non-tabular elements"""
import cv2
from table_blocks_det import utils

def elem_in_table(elem, table, thresh=4):
    """Check if element is inside a table.

        Args:
            elem (dict): element to check.
            table (dict): table to check within.
            thresh (int, optional): threshold between element and table.
                Defaults to 4. For lines thresh=10 is recommended.

        Returns:
            True (bool): if element is inside a table.
            False (bool): if element is not inside a table.

    """
    if (
            (table['x'] <= elem['x'] + thresh) and
            (table['x'] + table['w'] >= elem['x'] + elem['w'] - thresh) and
            (table['y'] <= elem['y'] + thresh) and
            (table['y'] + table['h'] >= elem['y'] + elem['h'] - thresh)
        ):
        return True
    return False

def elem_above_or_below_table(elem, table, thresh=4):
    """Check if element is inside a table.

        Args:
            elem (dict): element to check.
            table (dict): table to check.
            thresh (int, optional): threshold between element and table.
                Defaults to 4. For lines thresh=10 is recommended.

        Returns:
            'Above' (str): if element is above a table.
            'Below' (str): if element is below a table.

    """
    if table['y'] >= elem['y'] + elem['h'] - thresh:
        return 'Above'
    if table['y'] + table['h'] <= elem['y'] - thresh:
        return 'Below'

def table_by_lines(table, hor_lines, vert_lines, thresh=10):
    """Check if it is a table by containing lines.

        Args:
            table (dict): table to check
            hor_lines (list): horizontal lines
            vert_lines (list): vertical lines
            thresh (int, optional): threshold between line and table.
                Defaults to 10.

        Returns:
            True (bool): it is a table
            False (bool): it is not a table

    """
    num_hor, num_vert = 0, 0

    # Count lines inside a table
    for line in hor_lines:
        if elem_in_table(line, table, thresh=thresh):
            num_hor += 1
    for line in vert_lines:
        if elem_in_table(line, table, thresh=thresh):
            num_vert += 1

    # Table is considered as a table if:
    if (
            (num_hor == 2 and num_vert > 2) or
            (num_vert == 2 and num_hor > 2) or
            (num_hor > 2 and num_vert > 2) or
            (num_hor >= 3 and num_vert > 0) or
            (num_vert >= 3)
    ):
        return True
    return False

def set_textblocks_table_param(text_blocks, tables):
    for table in tables:
        for ind_row, row in enumerate(text_blocks):
            for ind_block, block in enumerate(row):
                if elem_in_table(block, table):
                    text_blocks[ind_row][ind_block]['in_table'] = True

def main(image_name, image_path):
    """Main script for detecting tables and non-table elements"""
    image = cv2.imread(image_path)

    # Load the data:
    folder_data = 'inv_processing/data'
    symbols = utils.load_data(f'{folder_data}/text_{image_name}.data')
    text_blocks = utils.load_data(f'{folder_data}/text_blocks_{image_name}.data')
    lines_dil = utils.load_data(f'{folder_data}/lines_dil_{image_name}.data')
    lines_vert = utils.load_data(f'{folder_data}/lines_vert_{image_name}.data')
    lines_hor = utils.load_data(f'{folder_data}/lines_hor_{image_name}.data')

    # Leave only narrow lines:
    for line in lines_vert[:]:
        if line['w'] >= line['h'] // 15:
            lines_vert.remove(line)
    for line in lines_hor[:]:
        if line['h'] >= line['w'] // 25:
            lines_hor.remove(line)

    img_h, img_w = image.shape[:2]
    # Additional parameters to text blocks:
    for ind_row, row in enumerate(text_blocks): # TODO: add these params from the beginning
        for ind_block in range(len(row)):
            text_blocks[ind_row][ind_block]['in_table'] = False
            text_blocks[ind_row][ind_block]['above_1'] = False
            text_blocks[ind_row][ind_block]['between'] = False
            text_blocks[ind_row][ind_block]['below_2'] = False
            text_blocks[ind_row][ind_block]['img_h'] = img_h
            text_blocks[ind_row][ind_block]['img_w'] = img_w
    # Additional parameters to symbols:
    for ind, _ in enumerate(symbols):
        symbols[ind]['in_table'] = False
        symbols[ind]['above_1'] = False
        symbols[ind]['between'] = False
        symbols[ind]['below_2'] = False
        symbols[ind]['img_h'] = img_h
        symbols[ind]['img_w'] = img_w

    tables = []
    # Find tables within a line_elements
    for line in lines_dil:
        if table_by_lines(line, lines_hor, lines_vert):
            tables.append(line)
    tables = sorted(tables, key=lambda table: table['y'])

    if tables:
        # Check if text_blocks and symbols in table:
        for table in tables:
            for ind_row, row in enumerate(text_blocks):
                for ind_block, block in enumerate(row):
                    if elem_in_table(block, table):
                        text_blocks[ind_row][ind_block]['in_table'] = True
            for ind, symbol in enumerate(symbols):
                if elem_in_table(symbol, table):
                    symbols[ind]['in_table'] = True

        # Save tables images:
        path_table_1 = f'results/{image_name}_tables.jpg'
        path_table_2 = f'results/{image_name}/{image_name}_tables.jpg'
        for path in (path_table_1, path_table_2):
            utils.bounding_boxes(image, image_name, method='row',
                                 row=tables, path=path)

    # Construct lists of non_table symbols and textblocks:
    symbols_non_table, text_blocks_non_table = [], []
    for symbol in symbols:
        if not symbol['in_table']:
            symbols_non_table.append(symbol)
    for row in text_blocks:
        for block in row:
            if not block['in_table']:
                text_blocks_non_table.append(block)

    # Additional params to text_blocks and symbols
    # Is it above 1st table, between tables, below 2nd table:
    if len(tables) > 1:
        table_1, table_2 = tables[0], tables[1]

        for ind, symbol in enumerate(symbols_non_table):
            if elem_above_or_below_table(symbol, table_1) == 'Above' and \
            elem_above_or_below_table(symbol, table_2) == 'Above':
                symbols_non_table[ind]['above_1'] = True
            elif elem_above_or_below_table(symbol, table_1) == 'Below' and \
                elem_above_or_below_table(symbol, table_2) == 'Above':
                symbols_non_table[ind]['between'] = True
            elif elem_above_or_below_table(symbol, table_1) == 'Below' and \
                elem_above_or_below_table(symbol, table_2) == 'Below':
                symbols_non_table[ind]['below_2'] = True

        for ind, block in enumerate(text_blocks_non_table):
            if elem_above_or_below_table(block, table_1) == 'Above' and \
            elem_above_or_below_table(block, table_2) == 'Above':
                text_blocks_non_table[ind]['above_1'] = True
            elif elem_above_or_below_table(block, table_1) == 'Below' and \
                elem_above_or_below_table(block, table_2) == 'Above':
                text_blocks_non_table[ind]['between'] = True
            elif elem_above_or_below_table(block, table_1) == 'Below' and \
                elem_above_or_below_table(block, table_2) == 'Below':
                text_blocks_non_table[ind]['below_2'] = True

    # Save images of non_table symbols and blocks:
    path_symbols_non_table = f'results/{image_name}/{image_name}_symbols_non_table.jpg'
    path_tb_non_table = f'results/{image_name}/{image_name}_blocks_non_table.jpg'
    utils.bounding_boxes(image, image_name, method='row',
                         row=symbols_non_table, path=path_symbols_non_table)
    utils.bounding_boxes(image, image_name, method='row',
                         row=text_blocks_non_table, path=path_tb_non_table)

    # Save non_table data:
    folder_non_table = 'data_not_in_table'
    utils.dump_data(data=symbols_non_table,
                    path=f'{folder_non_table}/symbols_non_table_{image_name}.data')
    utils.dump_data(data=text_blocks_non_table,
                    path=f'{folder_non_table}/blocks_non_table_{image_name}.data')

    # Save text blocks with new params:
    utils.dump_data(data=text_blocks,
                    path=f'{folder_data}/text_blocks_add_params_{image_name}.data')

if __name__ == "__main__":
    _IMAGE_NAME = "image_test"
    _IMAGE_PATH = "{}.jpg".format(_IMAGE_NAME)
    main(_IMAGE_NAME, _IMAGE_PATH)
