from copy import deepcopy as copy
import numpy as np
import cv2
from table_blocks_det import utils, cc_utils
# import utils, cc_utils
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image

THRESH = 3
SCALE = 100

class Cell:
    def __init__(self, bot_left, top_left, top_right, bot_right):
        """
        Corners coordinates: (x, y)
        """
        # Define corners:
        self.corners = dict()
        self.corners['bot_left'] = bot_left
        self.corners['bot_right'] = bot_right
        self.corners['top_left'] = top_left
        self.corners['top_right'] = top_right
        # Define borders:
        self.borders = dict()
        self.borders['left'] = (self.corners['top_left'], self.corners['bot_left'])
        self.borders['right'] = (self.corners['top_right'], self.corners['bot_right'])
        self.borders['top'] = (self.corners['top_left'], self.corners['top_right'])
        self.borders['bot'] = (self.corners['bot_left'], self.corners['bot_right'])
        # Define coords:
        self.row = None
        self.col = None
        self.row_span = None
        self.col_span = None

    def get_cell_image(self, image):
        pts = np.array([self.corners['bot_left'], self.corners['top_left'], self.corners['top_right'], self.corners['bot_right']], np.int32)
        # Crop the bnd rect:
        x, y, w, h = cv2.boundingRect(pts)
        cropped = image[y:y+h, x:x+w].copy()
        # Make mask:
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # Do bit-op:
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        # Add white background:
        bg = np.ones_like(cropped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        cell_image = dst + bg
        return cell_image

    def recognize_text(self, image):
        with PyTessBaseAPI(lang='rus', psm=PSM.SINGLE_BLOCK) as tesseract:
            cell_image = self.get_cell_image(image)
            img_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            tesseract.SetImage(img)
            text = tesseract.GetUTF8Text()
            return text

    def show_cell_on_table(self, table_image):
        image = copy(table_image)
        for bord in self.borders:
            p_1, p_2 = tuple(self.borders[bord][0]), tuple(self.borders[bord][1])
            cv2.line(image, p_1, p_2, (0, 255, 0), 2)
        utils.show_image(image)

class Table:
    # Threshold for rotated images (cells and etc.)
    THRESH = 3
    # Scale for lines morphological detection
    SCALE = 50

    def __init__(self, doc_image, image):
        self.doc_image = doc_image
        self.image = image

    def get_binary_image(self):
        thresh_image = utils.threshold_image(self.image)
        return 255 - thresh_image

    def get_mask(self):
        try:
            return self.mask
        except AttributeError:
            thresh_image = self.get_binary_image()
            horizontal_mask, vertical_mask = copy(thresh_image), copy(thresh_image)
            # Get horizontal lines mask:
            horizontal_size = int(self.doc_image.shape[1] / Table.SCALE)
            horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            horizontal_mask = cc_utils.highlight_lines(horizontal_mask, horizontal_structure)
            # Get vertical lines mask:
            vertical_size = int(self.doc_image.shape[0] / Table.SCALE) # TODO: mb diff scale for vert?
            vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
            vertical_mask = cc_utils.highlight_lines(vertical_mask, vertical_structure)
            # Dilate lines:
            kernel = np.ones((3, 3), np.uint8)
            horizontal_mask = cv2.dilate(horizontal_mask, kernel, iterations=1)
            vertical_mask = cv2.dilate(vertical_mask, kernel, iterations=1)
            # Combine to full table mask:
            self.mask = horizontal_mask + vertical_mask
            return self.mask

    def get_cells(self):
        try:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        except AttributeError:
            self.get_mask()
            contours, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for i in range(1, len(contours)):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            box = box.tolist()
            # Define corners:
            # Top left:
            top_left = box[0]
            for corner in box:
                if corner[0] + corner[1] < top_left[0] + top_left[1]:
                    top_left = corner
            box.remove(top_left)
            # Top right:
            top_right = box[0]
            for corner in box:
                if abs(corner[1] - top_left[1]) < abs(top_right[1] - top_left[1]):
                    top_right = corner
            box.remove(top_right)
            # Bot left
            bot_left = box[0]
            for corner in box:
                if corner[0] < bot_left[0]:
                    bot_left = corner
            box.remove(bot_left)
            # Bot right
            bot_right = box[0]
            cells.append(Cell(bot_left, top_left, top_right, bot_right))
            self.table = []
            # self.assign_cell_pos(cells)
            # cell_positions(cells, self.cells)
        cells.sort(key=lambda cell: cell.corners['top_left'][1])
        return cells

    def assign_cell_pos(self, cells, row=1, col=1):
        if not all([cell.row for cell in cells]):
            # Construct row:
            candidates = []
            # Choose top y cell:
            for cell in cells:
                if cell.row is None:
                    top_y_cell = cell
                    cell.row = row
                    candidates.append(top_y_cell)
                    break
            # Append other cells approximately in that row:
            for cell in cells:
                if cell.row is None:
                    delta = abs(top_y_cell.corners['top_left'][1] - cell.corners['top_left'][1])
                    # Choose candidates:
                    if delta < THRESH*6:
                        candidates.append(cell)
            candidates.sort(key=lambda candidate: candidate.corners['top_left'][0])
            self.table.append(candidates)
            for ind, cell in enumerate(candidates):
                cell.row = row
                cell.col = col + ind
            self.assign_cell_pos(cells, row=row+1, col=1)

def cell_positions(cells, table, row=1, col=1):
    if not all([cell.row for cell in cells]):
        # Construct row:
        candidates = []
        # Choose top y cell:
        for cell in cells:
            if cell.row is None:
                top_y_cell = cell
                cell.row = row
                candidates.append(top_y_cell)
                break
        # Append other cells approximately in that row:
        for cell in cells:
            if cell.row is None:
                delta = abs(top_y_cell.corners['top_left'][1] - cell.corners['top_left'][1])
                # Choose candidates:
                if delta < THRESH*6:
                    candidates.append(cell)
        candidates.sort(key=lambda candidate: candidate.corners['top_left'][0])
        table.append(candidates)
        for ind, cell in enumerate(candidates):
            cell.row = row
            cell.col = col + ind
        cell_positions(cells, table, row=row+1, col=1)

def cell_span(image, cells):
    for cell in cells:
        # Check how many cells on top:
        top_col_span = 0
        for other_cell in cells:
            if cell != other_cell:
                x1_l, x1_r = cell.corners['top_left'][0], cell.corners['top_right'][0]
                x2_l, x2_r = other_cell.corners['bot_left'][0], other_cell.corners['bot_right'][0]
                if abs(cell.corners['top_left'][1] - other_cell.corners['bot_left'][1]) < THRESH*4 and \
                   x2_r - x1_l > THRESH*4 and x1_r - x2_l > THRESH*4:
                       top_col_span += 1

        # Check how many cells on bot:
        bot_col_span = 0
        for other_cell in cells:
            if cell != other_cell:
                x1_l, x1_r = cell.corners['bot_left'][0], cell.corners['bot_right'][0]
                x2_l, x2_r = other_cell.corners['top_left'][0], other_cell.corners['top_right'][0]
                if abs(cell.corners['bot_left'][1] - other_cell.corners['top_left'][1]) < THRESH*4 and \
                   x2_r - x1_l > THRESH*4 and x1_r - x2_l > THRESH*4:
                       bot_col_span += 1
        cell.col_span = max(top_col_span, bot_col_span)

        # Check how many cells to the left:
        left_row_span = 0
        for other_cell in cells:
            if cell != other_cell:
                y1_t, y1_b = cell.corners['top_left'][1], cell.corners['bot_left'][1]
                y2_t, y2_b = other_cell.corners['top_right'][1], other_cell.corners['bot_right'][1]
                if abs(cell.corners['top_left'][0] - other_cell.corners['top_right'][0]) < THRESH*4 and \
                   y1_b - y2_t > THRESH*4 and y2_b - y1_t > THRESH*4:
                       left_row_span += 1
        # Check how many cells to the right:
        right_row_span = 0
        for other_cell in cells:
            if cell != other_cell:
                y1_t, y1_b = cell.corners['top_right'][1], cell.corners['bot_right'][1]
                y2_t, y2_b = other_cell.corners['top_left'][1], other_cell.corners['bot_left'][1]
                if abs(cell.corners['top_right'][0] - other_cell.corners['top_left'][0]) < THRESH*4 and \
                   y1_b - y2_t > THRESH*4 and y2_b - y1_t > THRESH*4:
                       right_row_span += 1
        cell.row_span = max(left_row_span, right_row_span)

def generate_html_table(table, image):
    html_table = '<table>\n'
    for row in table:
        html_table += '  <tr>\n'
        for cell in row:
            text = cell.recognize_text(image)
            html_table += f'    <td colspan="{cell.col_span}" rowspan="{cell.row_span}">{text}</td>\n'
        html_table += '  </tr>\n'
    html_table += '</table>'
    return html_table


def main(doc_image, table_image):
    # orig_image = cv2.imread('image_test.jpg')
    # image = cv2.imread(image_path)

    # tab = Table(orig_image, image)
    tab = Table(doc_image, table_image)
    mmmask = tab.get_mask()

    # intersections = cv2.bitwise_and(horizontal_mask, vertical_mask)
    # utils.show_image(intersections)
    cells = tab.get_cells()

    table = []
    cell_positions(cells, table)
    cells.sort(key = lambda cell: (cell.row, cell.col))

    cell_span(table_image, cells)

    # for row in table:
    #     for cell in row:
    #         recognize_text(cell, image)
    #         show_cell(image, cell)
    # for row in table:
    #     for cell in row:
    #         print(f'Pos: {cell.row, cell.col}\nRow span: {cell.row_span}, Col span: {cell.col_span}')
    #         cell.show_cell_on_table(image)

    # tab.cells = table
    # html_table = generate_html_table(table, image)
    # print(html_table)

    tab.table = table

    return tab








def extract_table_mask(full_image, table_image):
    thresh_image = utils.threshold_image(table_image)
    thresh_image = 255 - thresh_image
    # utils.show_image(thresh)

    scale = 100
    horizontal_mask, vertical_mask = copy(thresh_image), copy(thresh_image)

    horizontal_size = int(full_image.shape[1] / scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = cc_utils.highlight_lines(horizontal_mask, horizontal_structure)
    # utils.show_image(horizontal_mask)

    vertical_size = int(full_image.shape[0] / scale) # TODO: mb diff scale for vert?
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = cc_utils.highlight_lines(vertical_mask, vertical_structure)
    # utils.show_image(vertical_mask)

    # Dilate lines:
    kernel = np.ones((3, 3), np.uint8)
    horizontal_mask = cv2.dilate(horizontal_mask, kernel, iterations=1)
    vertical_mask = cv2.dilate(vertical_mask, kernel, iterations=1)

    mask = horizontal_mask + vertical_mask
    return mask

def show_cell_deb(image, cell, other_cell):
    image = copy(image)
    for bord in cell.borders:
        p_1, p_2 = tuple(cell.borders[bord][0]), tuple(cell.borders[bord][1])
        cv2.line(image, p_1, p_2, (0, 255, 0), 2)
    for bord in other_cell.borders:
        p_1, p_2 = tuple(other_cell.borders[bord][0]), tuple(other_cell.borders[bord][1])
        cv2.line(image, p_1, p_2, (255, 0, 0), 2)
    utils.show_image(image)

if __name__ == "__main__":
    # image_name = 'table'
    image_path = 'table2.jpg'
    main(image_path)
