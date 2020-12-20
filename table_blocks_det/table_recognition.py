from copy import deepcopy as copy
import numpy as np
import cv2
from inv_processing import cc_utils, utils

THRESH = 3

def highlight_lines(mask, struct_el):
    lines_mask = cv2.erode(mask, struct_el) # erosion: object size decreases
    lines_mask = cv2.dilate(lines_mask, struct_el) # dilation: object size increases
    # lines_mask = cv2.dilate(lines_mask, struct_el) # dilation: object size increases
    return lines_mask

def show_cell(image, cell):
    image = copy(image)
    for bord in cell.borders:
        p_1, p_2 = tuple(cell.borders[bord][0]), tuple(cell.borders[bord][1])
        cv2.line(image, p_1, p_2, (0, 255, 0), 2)
    utils.show_image(image)

def show_cell_deb(image, cell, other_cell):
    image = copy(image)
    for bord in cell.borders:
        p_1, p_2 = tuple(cell.borders[bord][0]), tuple(cell.borders[bord][1])
        cv2.line(image, p_1, p_2, (0, 255, 0), 2)
    for bord in other_cell.borders:
        p_1, p_2 = tuple(other_cell.borders[bord][0]), tuple(other_cell.borders[bord][1])
        cv2.line(image, p_1, p_2, (255, 0, 0), 2)
    utils.show_image(image)

def main(image_name, image_path):
    orig_image = cv2.imread('image_test.jpg')
    image = cv2.imread(image_path)

    thresh_image = utils.threshold_image(image)
    thresh_image = 255 - thresh_image
    # utils.show_image(thresh)

    scale = 100
    horizontal_mask, vertical_mask = copy(thresh_image), copy(thresh_image)

    horizontal_size = int(orig_image.shape[1] / scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = highlight_lines(horizontal_mask, horizontal_structure)
    # utils.show_image(horizontal_mask)

    vertical_size = int(orig_image.shape[0] / scale) # TODO: mb diff scale for vert?
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = highlight_lines(vertical_mask, vertical_structure)
    # utils.show_image(vertical_mask)

    # Dilate lines:
    kernel = np.ones((3, 3), np.uint8)
    horizontal_mask = cv2.dilate(horizontal_mask, kernel, iterations=1)
    vertical_mask = cv2.dilate(vertical_mask, kernel, iterations=1)

    mask = horizontal_mask + vertical_mask
    # utils.show_image(mask)

    # intersections = cv2.bitwise_and(horizontal_mask, vertical_mask)
    # utils.show_image(intersections)

    # Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
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

    cells.sort(key=lambda cell: cell.corners['top_left'][1])

    cell_positions(image, cells)

    cells.sort(key = lambda cell: (cell.row, cell.col))

    cell_span(image, cells)

    # for cell in cells:
    #     print(f'Pos: {cell.row, cell.col}\nRow span: {cell.row_span}, Col span: {cell.col_span}')
    #     show_cell(image, cell)

def cell_positions(image, cells, row=1, col=1):
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
        for ind, cell in enumerate(candidates):
            cell.row = row
            cell.col = col + ind
        cell_positions(image, cells, row=row+1, col=1)

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

class Cell:
    def __init__(self, bot_left, top_left, top_right, bot_right):
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

if __name__ == "__main__":
    image_name = 'table'
    image_path = 'table1.jpg'
    main(image_name, image_path)
