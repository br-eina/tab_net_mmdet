import cv2
from collections import Counter
import pickle
import csv
import os

def show_image(image):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)

def save_image(image, filename):
    cv2.imwrite(filename, image)

def sort_rows(rows):
    sorted_rows = []
    for row in rows:
        row = sorted(row, key = lambda el: (el['x'], el['y']))
        sorted_rows.append(row)
    return sorted_rows

def show_lines(image, lines_elem, image_name):
    image_to_show = image.copy()
    for element in lines_elem:
        p_1 = (element['x'], element['y'])
        p_2 = (element['x'] + element['w'], element['y'] + element['h'])
        cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 3)
    save_image(image_to_show, "table_blocks_det/results/lines/lines_{}.jpg".format(image_name))

def show_bbox(image, element):
    image_to_show = image.copy()
    p_1 = (element['x'], element['y'])
    p_2 = (element['x'] + element['w'], element['y'] + element['h'])
    cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 5)
    show_image(image_to_show)

def show_text_elem_in_roi(image, text_elem_in_roi):
    image_to_show = image.copy()
    for element in text_elem_in_roi:
        p_1 = (element['x'], element['y'])
        p_2 = (element['x'] + element['w'], element['y'] + element['h'])
        cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 5)
    show_image(image_to_show)

def show_text_blocks_in_row(image, rows, image_name):
    image_to_show = image.copy()
    for row in rows:
        for element in row:
            p_1 = (element['x'], element['y'])
            p_2 = (element['x'] + element['w'], element['y'] + element['h'])
            cv2.rectangle(image_to_show, p_1, p_2, (0, 0, 255), 4)
    save_image(image_to_show, "table_blocks_det/results/{0}_blocks.jpg".format(image_name))
    save_image(image_to_show, "table_blocks_det/results/{0}/{0}_blocks.jpg".format(image_name))

def detach_blocks_by_vert_lines(element_1, element_2, vert_lines):
    key = 0
    for line in vert_lines:
        if (
            ((element_1['x'] + element_1['w']) <= line['x'] <= element_2['x']) and
            (line['y'] <= element_1['y']) and
            ((line['y'] + line['h']) >= (element_1['y'] + element_1['h']))
        ):
            key += 1
    if key != 0:
        return 1
    else:
        return 0

def save_text_blocks(text_blocks_rows, image_name):
    with open("table_blocks_det/data/text_blocks_{0}.data".format(image_name), 'wb') as outfile:
        pickle.dump(text_blocks_rows, outfile)

# def save_cropped_blocks(image, text_blocks_rows, image_name):
#     if not os.path.exists('images/cropped_blocks/{0}'.format(image_name)):
#         os.makedirs('images/cropped_blocks/{0}'.format(image_name))
#     for ind_row, blocks_row in enumerate(text_blocks_rows):
#         for ind_block, text_block in enumerate(blocks_row):
#             t = 3
#             y = text_block['y']
#             if y >= t:
#                 y = text_block['y'] - t
#             h = text_block['h'] + 2*t
#             x = text_block['x']
#             if x >= t:
#                 x = text_block['x'] - t
#             w = text_block['w'] + 2*t
#             cropped_image = image[y:y+h, x:x+w].copy()
#             filename = "images/cropped_blocks/{0}/{1}-{2}.jpg".format(image_name, ind_row, ind_block)
#             save_image(cropped_image, filename)

def main(image_name, image_path):
    image = cv2.imread(image_path)

    # Load text elem
    with open('table_blocks_det/data/rows_ccomp_{0}.data'.format(image_name), 'rb') as filehandle:
        # read the data as binary data stream
        rows = pickle.load(filehandle)

    # Load vertical lines elem
    with open('table_blocks_det/data/lines_vert_{0}.data'.format(image_name), 'rb') as filehandle:
        # read the data as binary data stream
        vert_lines = pickle.load(filehandle)

    all_distances_rows = []
    # Read all distances csv
    with open('table_blocks_det/data/distance_data.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            all_distances_rows.append(row)
    csvFile.close()

    # Make single all distances list:
    all_distances = []
    for d_row in all_distances_rows:
        for dist in d_row:
            all_distances.append(dist)

    count_all_distances = Counter(all_distances).most_common()

    # Distances with count 1000 + :
    dist_large_count = []
    for dist_tuple in count_all_distances:
        if dist_tuple[1] >= 200:
            dist_large_count.append(int(dist_tuple[0]))

    # Calculate distances between consecutive characters
    dist_doc_rows = []
    for index_row, row in enumerate(rows):
        dist_doc_rows.append([])
        for i in range(len(row)):
            next_i = i + 1
            if i != len(row) - 1:
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
            dist_doc_rows[index_row].append(d)

    # Add distance parameter to elements
    for ind_row, d_row in enumerate(dist_doc_rows):
        for ind_elem, distance in enumerate(d_row):
            rows[ind_row][ind_elem]['d_next'] = distance


    # Make single dist_doc list:
    dist_doc = []
    for d_row in dist_doc_rows:
        for dist in d_row:
            dist_doc.append(dist)

    not_consec_dist = []
    # Removing distances for consecutive characters and words:
    for dist in dist_doc:
        if dist not in dist_large_count:
            not_consec_dist.append(dist)

    avg_dist_characters = 2
    avg_dist_words = 14

    # Construct text blocks:
    # Задаем массивы y_top ; y_bot - границы блока по макс-мин букве
    y_top_list = []
    y_bot_list = []
    text_blocks_rows = []
    for ind_row, row in enumerate(rows):
        # Начальные координаты первого блока:
        x_l = row[0]['x']
        y_t = row[0]['y']
        text_blocks_rows.append([])
        chars = 0
        for ind_elem, elem in enumerate(row):
            # Заполняем массив по каждому элементу
            y_top = elem['y']
            y_top_list.append(y_top)
            y_bot = elem['y'] + elem['h']
            y_bot_list.append(y_bot)

            chars += 1

            ind_next_elem = ind_elem + 1
            # Если элемент в массиве не последний:
            if ind_elem != len(row) - 1:
                detach_blocks = detach_blocks_by_vert_lines(row[ind_elem], row[ind_next_elem], vert_lines)
                # Если расст до след элемента не входит в разграничивающие расстояния:
                if (
                    ((elem['d_next'] not in not_consec_dist) or
                    (0 <= elem['d_next'] <= avg_dist_characters * 3) or
                    (avg_dist_words / 2 <= elem['d_next'] <= avg_dist_words * 3)) and
                    detach_blocks == 0
                ):
                    # Определяем конечные координаты элемента:
                    x_r = elem['x'] + elem['w']
                    y_top = elem['y']
                    y_top_list.append(y_top)
                    y_t = min(y_top_list) # текущая y_top коорд блока
                    y_bot = elem['y'] + elem['h']
                    y_bot_list.append(y_bot)
                    y_b = max(y_bot_list) # текущая y_bot коорд блока

                # Если нужно разграничить блоки:
                else:
                    # Определяем конечные координаты элемента:
                    x_r = elem['x'] + elem['w']
                    y_top = elem['y']
                    y_top_list.append(y_top)
                    y_t = min(y_top_list) # текущая y_top коорд блока
                    y_bot = elem['y'] + elem['h']
                    y_bot_list.append(y_bot)
                    y_b = max(y_bot_list) # текущая y_bot коорд блока
                    w = x_r - x_l
                    h = y_b - y_t
                    # Задаем текстовый блок:
                    text_block = {}
                    text_block['x'] = x_l
                    text_block['y'] = y_t
                    text_block['w'] = w
                    text_block['h'] = h
                    text_block['chars'] = chars
                    text_block['in_table'] = False
                    text_blocks_rows[ind_row].append(text_block)

                    # Обнуляем массивы y_top ; y_bot
                    y_top_list = []
                    y_bot_list = []
                    chars = 0

                    # Задаем начальные координаты следующего блока:
                    next_element = row[ind_next_elem]
                    x_l = next_element['x']
                    y_t = next_element['y']
            # Если элемент в массиве последний, то закрываем блок:
            else:
                # Определяем конечные координаты элемента:
                x_r = elem['x'] + elem['w']
                y_top = elem['y']
                y_top_list.append(y_top)
                y_t = min(y_top_list) # текущая y_top коорд блока
                y_bot = elem['y'] + elem['h']
                y_bot_list.append(y_bot)
                y_b = max(y_bot_list) # текущая y_bot коорд блока
                w = x_r - x_l
                h = y_b - y_t
                # Задаем текстовый блок:
                text_block = {}
                text_block['x'] = x_l
                text_block['y'] = y_t
                text_block['w'] = w
                text_block['h'] = h
                text_block['chars'] = chars
                text_block['in_table'] = False
                text_blocks_rows[ind_row].append(text_block)

                # Обнуляем массивы y_top ; y_bot
                y_top_list = []
                y_bot_list = []
                chars = 0

    show_text_blocks_in_row(image, text_blocks_rows, image_name)
    save_text_blocks(text_blocks_rows, image_name)

if __name__ == "__main__":
    _image_name = "image_test"
    _image_path = "{}.jpg".format(_image_name)
    main(_image_name, _image_path)
