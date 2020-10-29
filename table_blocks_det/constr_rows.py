import cv2
from copy import deepcopy as copy
import random
from collections import Counter
import pickle
from table_blocks_det import utils

def show_lines(image, lines_elem):
    image_to_show = image.copy()
    for element in lines_elem:
        p_1 = (element['x'], element['y'])
        p_2 = (element['x'] + element['w'], element['y'] + element['h'])
        cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), -1)
        utils.show_image(image_to_show)

def show_bbox(image, element):
    image_to_show = image.copy()
    p_1 = (element['x'], element['y'])
    p_2 = (element['x'] + element['w'], element['y'] + element['h'])
    cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 3)
    utils.show_image(image_to_show)

def show_two_bbox(image, element1, element2):
    image_to_show = image.copy()
    p_1 = (element1['x'], element1['y'])
    p_2 = (element1['x'] + element1['w'], element1['y'] + element1['h'])
    p_3 = (element2['x'], element2['y'])
    p_4 = (element2['x'] + element2['w'], element2['y'] + element2['h'])
    cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 3)
    cv2.rectangle(image_to_show, p_3, p_4, (0, 0, 255), 3)
    utils.show_image(image_to_show)

def show_text_elem_in_roi(image, text_elem_in_roi):
    image_to_show = image.copy()
    for element in text_elem_in_roi:
        p_1 = (element['x'], element['y'])
        p_2 = (element['x'] + element['w'], element['y'] + element['h'])
        cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 5)
    utils.show_image(image_to_show)

def show_conn_comp_in_row(image, rows, image_name):
    image_to_show = image.copy()
    for row in rows:
        b = random.randint(0, 255)
        q = random.randint(0, 255)
        r = random.randint(0, 255)
        for element in row:
            p_1 = (element['x'], element['y'])
            p_2 = (element['x'] + element['w'], element['y'] + element['h'])
            cv2.rectangle(image_to_show, p_1, p_2, (b, q, r), 2)
    utils.save_image(image_to_show, "table_blocks_det/results/{0}/{0}_char_rows.jpg".format(image_name))

def save_rows(rows, image_name):
    with open("table_blocks_det/data/rows_ccomp_{0}.data".format(image_name), 'wb') as outfile:
        pickle.dump(rows, outfile)


def check_if_el_in_roi(elem, roi):
    if (
        elem['x'] >= roi['x'] and elem['y'] >= roi['y'] and
        (elem['x'] + elem['w']) <= (roi['x'] + roi['w']) and
        (elem['y'] + elem['h']) <= (roi['y'] + roi['h'])
    ):
        return 1
    else:
        return 0

def check_if_two_el_in_row(elem_1, elem_2):
    t = 2
    y1_top = elem_1['y']
    y1_bot = y1_top + elem_1['h']
    y2_top = elem_2['y']
    y2_bot = y2_top + elem_2['h']
    if (y1_top <= y2_bot - t and y1_bot >= y2_top + t) or (y2_top <= y1_bot - t and y2_bot >= y1_top + t):
        return 1
    else:
        return 0

def check_if_two_el_to_connect(elem_1, elem_2, row_dist):
    x1_l = elem_1['x']
    x1_r = x1_l + elem_1['w']
    x2_l = elem_2['x']
    x2_r = x2_l + elem_2['w']

    y1_t = elem_1['y']
    y1_b = y1_t + elem_1['h']
    y2_t = elem_2['y']
    y2_b = y2_t + elem_2['h']

    A_1 = (x1_r - x1_l) * (y1_b - y1_t)
    A_2 = (x2_r - x2_l) * (y2_b - y2_t)

    cond_2_in_1 = 0
    cond_1_in_2 = 0
    cond_overlap = 0
    cond_adjoin = 0

    SI = max(0, min(x1_r, x2_r) - max(x1_l, x2_l)) * max(0, min(y1_b, y2_b) - max(y1_t, y2_t))
    if SI > 0:
        cond_overlap = 1

    if A_1 / A_2 <= 0.3 or A_2 / A_1 <= 0.3:

        if (x2_l >= x1_l or abs(x2_l - x1_l) <= 3) and (x2_r <= x1_r or abs(x2_r - x1_r) <= 3) and (abs(y1_t - y2_b) <= row_dist // 6 or abs(y2_t - y1_b) <= row_dist // 6):
            cond_2_in_1 = 1
        if (x1_l >= x2_l or abs(x1_l - x2_l) <= 3) and (x1_r <= x2_r or abs(x1_r - x2_r) <= 3) and (abs(y2_t - y1_b) <= row_dist // 6 or abs(y1_t - y2_b) <= row_dist // 6):
            cond_1_in_2 = 1

    if cond_1_in_2 == 1 or cond_2_in_1 == 1 or cond_overlap == 1 or cond_adjoin == 1:
        return 1
    else:
        return 0

def construct_bbox(elem_1, elem_2):
    x1_l = elem_1['x']
    x1_r = x1_l + elem_1['w']
    x2_l = elem_2['x']
    x2_r = x2_l + elem_2['w']

    y1_t = elem_1['y']
    y1_b = y1_t + elem_1['h']
    y2_t = elem_2['y']
    y2_b = y2_t + elem_2['h']

    x_l = min(x1_l, x2_l)
    y_t = min(y1_t, y2_t)
    x_r = max(x1_r, x2_r)
    y_b = max(y1_b, y2_b)

    elem = {}
    elem['x'] = x_l
    elem['y'] = y_t
    elem['w'] = x_r - x_l
    elem['h'] = y_b - y_t

    return elem

# Добавляем текстовые элементы в ряд
def add_elem_in_row(added_elem_list, remain_text, rows, row_ind):
    num_add_elem = 0
    for added_elem in added_elem_list[:]:
        for text_elem in remain_text[:]:
            if check_if_two_el_in_row(added_elem, text_elem) == 1:
                added_elem_list.append(text_elem)
                rows[row_ind].append(text_elem)
                remain_text.remove(text_elem)
                num_add_elem += 1
    if num_add_elem > 0:
        add_elem_in_row(added_elem_list, remain_text, rows, row_ind)

def add_if_condition(consid_elem, remain_rows, new_rows, row_ind, mean_dist):
    elem_add = 0
    for remain_elem in remain_rows[row_ind][:]:
        if check_if_two_el_to_connect(remain_elem, consid_elem, mean_dist):
            new_elem = construct_bbox(remain_elem, consid_elem)
            new_rows[row_ind].remove(consid_elem)
            new_rows[row_ind].append(new_elem)
            remain_rows[row_ind].remove(remain_elem)
            elem_add += 1
            break
    if elem_add > 0:
        consid_elem = new_elem
        add_if_condition(consid_elem, remain_rows, new_rows, row_ind, mean_dist)

def add_elem_with_condit(consid_elem, remain_rows, new_rows, row_ind, mean_dist):
    key = 0
    for remain_elem in remain_rows[row_ind][:]:
        # Если подходит под условие, то удаляем из рассм и добавляем в новый
        if check_if_two_el_to_connect(remain_elem, consid_elem, mean_dist):
            new_elem = construct_bbox(remain_elem, consid_elem)
            new_rows[row_ind].append(new_elem)
            remain_rows[row_ind].remove(remain_elem)
            key += 1
            break
    # Смотрим, есть ли для добавленного элемента еще перекрывающие эл-ты:
    if key > 0:
        consid_elem = new_elem
        add_if_condition(consid_elem, remain_rows, new_rows, row_ind, mean_dist)

    # Если нет элемента под условие, то добавляем рассматриваемый эл-т:
    if key == 0:
        new_rows[row_ind].append(consid_elem)
        # Выбираем новый рассм элемент:
        if len(remain_rows[row_ind]) != 0:
            consid_elem = remain_rows[row_ind][0]
            remain_rows[row_ind].remove(consid_elem)
            add_elem_with_condit(consid_elem, remain_rows, new_rows, row_ind, mean_dist)
    # Если эл-т с условием был и добавили в новый ряд, то выбираем новый рассм эл-т:
    else:
        if len(remain_rows[row_ind]) != 0:
            consid_elem = remain_rows[row_ind][0]
            remain_rows[row_ind].remove(consid_elem)
            add_elem_with_condit(consid_elem, remain_rows, new_rows, row_ind, mean_dist)

def check_if_line_in_roi(elem, roi, avg_height):
    if (
        (((elem['x'] <= roi['x']) and ((elem['x'] + elem['w']) >= roi['x'])) or (roi['x'] <= elem['x'] <= (roi['x'] + roi['w']))) and
        elem['y'] >= roi['y'] + avg_height and (elem['y'] + elem['h']) <= (roi['y'] + roi['h'])
    ):
        return 1
    else:
        return 0

def main(image_name, image_path):

    image = cv2.imread(image_path)
    data_folder = 'table_blocks_det/data'

    # Load the data
    with open(f'{data_folder}/text_{image_name}.data', 'rb') as datafile:
        text_elements = pickle.load(datafile)
    with open(f'{data_folder}/avg_h_{image_name}.data', 'rb') as datafile:
        avg_height = pickle.load(datafile)
    with open(f'{data_folder}/lines_hor_{image_name}.data', 'rb') as datafile:
        hor_lines = pickle.load(datafile)

    # Construct lines from text elem in:
    remain_text = copy(text_elements)
    rows = [[] for _ in range(len(text_elements) // 2)]

    rows[0].append(remain_text[0])
    remain_text.remove(remain_text[0])

    # Construct rows from text elements:
    for row_ind, row in enumerate(rows):
        added_elem_list = []
        rows[row_ind].append(remain_text[0]) # Add first remaining element into new row
        added_elem_list.append(remain_text[0])
        remain_text.remove(remain_text[0]) # Remove added element from remaining elements
        if len(remain_text) == 0:
            break
        add_elem_in_row(added_elem_list, remain_text, rows, row_ind)
        if len(remain_text) == 0:
            break

    rows = utils.remove_empty_rows(rows)

    # Find row positions:
    row_positions = []
    for ind_row, row in enumerate(rows):
        mass_y_top = []
        mass_y_bot = []
        for elem in row:
            mass_y_top.append(elem['y'])
            mass_y_bot.append(elem['y'] + elem['h'])
        y_top = min(mass_y_top)
        y_bot = max(mass_y_bot)

        row_positions.append({
            'ind': ind_row,
            'y_top': y_top,
            'y_bot': y_bot
        })

    sorted_row_pos = sorted(row_positions, key=lambda k: k['y_top'])

    # Calculate distances between rows:
    distances_between_rows = []
    for ind_row, pos in enumerate(sorted_row_pos):
        if ind_row != len(sorted_row_pos) - 1:
            distance = abs(sorted_row_pos[ind_row + 1]['y_top'] - pos['y_bot'])
            distances_between_rows.append(distance)

    # Most common distance
    common_dist = Counter(distances_between_rows).most_common(1)

    common_dist = common_dist[0][0]

    # Mean distance:
    sum_dist = sum(distances_between_rows)
    mean_dist = sum_dist // len(distances_between_rows)

    # Connect small horizontal elements:
    remain_rows = copy(rows)
    new_rows = [[] for _ in range(len(rows))]

    # Формируем новый ряд из текстовых элементов (соединенные малые части):
    for row_ind, row in enumerate(new_rows):
        # Если длина оставшегося ряда = 0, то прекращаем
        if len(remain_rows[row_ind]) == 0:
            break

        consid_elem = remain_rows[row_ind][0]
        remain_rows[row_ind].remove(consid_elem)
        add_elem_with_condit(consid_elem, remain_rows, new_rows, row_ind, mean_dist)

    rows = utils.sort_rows_x_y(new_rows)

    # Find row positions:
    row_positions = []
    for ind_row, row in enumerate(rows):
        mass_y_top = []
        mass_y_bot = []
        mass_x_left = []
        mass_x_right = []
        for elem in row:
            mass_x_left.append(elem['x'])
            mass_x_right.append(elem['x'] + elem['w'])
            mass_y_top.append(elem['y'])
            mass_y_bot.append(elem['y'] + elem['h'])
        x_left = min(mass_x_left)
        x_right = max(mass_x_right)
        y_top = min(mass_y_top)
        y_bot = max(mass_y_bot)

        row_positions.append({
            'ind': ind_row,
            'x': x_left,
            'y': y_top,
            'w': x_right - x_left,
            'h': y_bot - y_top

        })

    new_rows = []

    # Divide rows by horizontal lines:
    new_rows = []
    for ind_row, row_position in enumerate(row_positions):
        for line in hor_lines:
            if check_if_line_in_roi(line, row_position, avg_height) == 1:
                row_under_line = []
                trigger = 0
                for element in rows[ind_row][:]:
                    if (
                        (element['x'] >= line['x']) and ((element['x'] + element['w']) <= (line['x'] + line['w'])) and
                        (element['y'] >= line['y'])
                    ):
                        trigger += 1
                        row_under_line.append(element)
                        rows[ind_row].remove(element)

                if trigger != 0:
                    new_rows.append(row_under_line)

    if len(new_rows) != 0:
        rows = rows + new_rows

    rows = utils.sort_rows_x_y(rows)

    show_conn_comp_in_row(image, rows, image_name)

    # Calculate distances between consecutive characters
    d_list = []
    for index_row, row in enumerate(rows):
        d_list.append([])
        for i in range(len(row)):
            next_i = i + 1
            if next_i < len(row):
                element = row[i]
                next_element = row[next_i]
                x_left_1 = element['x']
                x_right_1 = x_left_1 + element['w']
                x_left_2 = next_element['x']
                x_right_2 = x_left_2 + next_element['w']
                d_12 = abs(x_left_1 - x_right_2)
                d_21 = abs(x_left_2 - x_right_1)
                d = min(d_12, d_21)
            else:
                d = 0
            d_list[index_row].append(d)

    # Make single d list
    d_single_list = []
    for d_row in d_list:
        for dist in d_row:
            d_single_list.append(dist)

    save_rows(rows, image_name)

if __name__ == "__main__":
    _image_name = "image_test"
    _image_path = "{}.jpg".format(_image_name)
    main(_image_name, _image_path)
