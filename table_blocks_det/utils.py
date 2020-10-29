"""Provides high-level support for CV algorithms and docs processing."""
from random import randint as rand
from os import makedirs
from os.path import exists
import pickle
import cv2



def show_image(image):
    """Visualize image in the floating window.

    Args:
        image (np.array): image to visualize.

    Returns:
        None

    """
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)

def save_image(image, path):
    """Save image in the specialized folder.

    Args:
        image (np.array): image to save
        path (str): image path

    Returns:
        None

    """
    cv2.imwrite(path, image)

def grayscale_image(image):
    """Grayscale the image.

    Args:
        image (np.array): image to grayscale

    Returns:
        gray_image (np.array): grayscaled image

    """
    num_channels = 3
    if len(image.shape) == num_channels:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def threshold_image(image):
    """Threshold the image (OTSU).

    Args:
        image (np.array): image to threshold

    Returns:
        thresh_image (np.array): thresholded image

    """
    gray = grayscale_image(image)
    thresh_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh_image

def remove_empty_rows(rows):
    """Remove almost empty (len(row) <= 1) row from a list of rows

        Args:
            rows (list): list of rows to process

        Returns:
            processed_rows (list): non-empty rows

    """
    processed_rows = []
    for row in rows:
        if len(row) > 1:
            processed_rows.append(row)
    return processed_rows

def sort_rows_x_y(rows):
    """Sort rows by 'x' and by 'y'

        Args:
            rows (list): rows to sort

        Returns:
            sorted_rows (list): sorted rows

    """
    sorted_rows = []
    for row in rows:
        row = sorted(row, key=lambda el: (el['x'], el['y']))
        sorted_rows.append(row)
    return sorted_rows

def flatten_list(nested_list):
    """Flatten nested list (list of lists)

        Args:
            nested_list (list): list to flatten. Dim = 2.

        Returns:
            flat_list (list): flattened nested_list.

    """
    return [element for sublist in nested_list for element in sublist]

def draw_single_bbox(image, element, color=(0, 0, 255), width=3):
    """Draw bounding box of a single element. Changes the input image.

        Args:
            image (np.array): background image to draw bbox on.
            element (dict): element around which the bbox is constructed.
            color (tuple, optional): color (B, G, R) of bbox. Defaults to (0, 0, 255): red.
            width (int, optional): width of bbox. Defaults to 3.

        Returns:
            None

    """
    p_1 = (element['x'], element['y'])
    p_2 = (element['x'] + element['w'], element['y'] + element['h'])
    cv2.rectangle(image, p_1, p_2, color, width)

def bounding_boxes(image, image_name, method, element=None, row=None, rows=None, label_field=None,
                   folder=None, path=None, save=True, show=False, color=(0, 0, 255), width=3):
    """Save or show bounding boxes around desired elements.

    Elements data structure should be a dict{x: val, y: val, w: val, h: val} or list of such dicts.
    To construct a bbox around one element -> define element.
    To construct bboxes around elements in a single row -> define row.
    To construct bboxes row by row in a list of rows -> define rows.

        Args:
            image (np.array): background image to draw bboxes on.
            image_name (str): image name.
            method (str): type of incoming data.
                'elem' if bbox is around one element;
                'row' if bboxes are around each element in a single row;
                'by_row' if bboxes are around each element row by row in rows.
            element (dict, optional): define if method == 'elem'. Defaults to None.
            row (list, optional): define if method == 'row'. Defaults to None.
            rows (list, optional): define if method == 'by_row'. Defaults to None.
            label_field (str, optional): draw label or predictions around label_field if defined. Define rows.
                Defaults to None.
            folder (str, optional): specified folder to save image in.
                Defaults to 'results/{image_name}/table_debug'
            path (str, optional): full path of the saved image. Defaults to None.
            save (bool, optional): save image if True. Defaults to True.
            show (bool, optional): show image if True. Defaults to False.
            color (tuple, optional): color (B, G, R) of bbox. Defaults to (0, 0, 255): red.
            width (int, optional): width of bbox. Defaults to 3.

        Returns:
            image_bbox (np.array): image with bboxes.

     """
    image_bbox = image.copy()
    if not folder:
        folder = f'results/{image_name}/table_debug'
    if not label_field:
        if method == 'elem':
            draw_single_bbox(image_bbox, element, color, width)
        elif method == 'row':
            for element in row:
                draw_single_bbox(image_bbox, element, color, width)
        elif method == 'by_row':
            for row in rows:
                blue, green, red = rand(0, 255), rand(0, 255), rand(0, 255)
                color = (blue, green, red)
                for element in row:
                    draw_single_bbox(image_bbox, element, color, width)
                if show:
                    show_image(image_bbox)
        else:
            raise ValueError("Method must be 'elem', 'row' or 'by_row'")
    else:
        for element in row:
            if element[label_field] == 'misc':
                continue
            if element[label_field] == 'doc_id':
                color = (34, 34, 178)
            elif element[label_field] == 'info':
                color = (80, 127, 255)
            elif element[label_field] == 'total':
                color = (34, 139, 34)
            elif element[label_field] == 'verif':
                color = (211, 0, 148)
            draw_single_bbox(image_bbox, element, color)
    if save:
        if path:
            # Create folders if necessary
            ind_end_folder = len(path) - path[::-1].find('/')
            folder_in_path = path[:ind_end_folder]
            create_folders(folder_in_path)
            save_image(image_bbox, path)
        else:
            save_image(image_bbox, f'{folder}/{image_name}_{method}.jpg')
    if show:
        show_image(image_bbox)
    return image_bbox

def dump_data(data, path):
    """Dumps data. Path must be specified.

        Args:
            data (data): data to dump.
            path (str): desired path of the data.

        Returns:
            None

    """
    with open(path, 'wb') as datafile:
        pickle.dump(data, datafile)

def load_data(path):
    """Loads data. Path must be specified.

        Args:
            path (str): path of the data.

        Returns:
            data

    """
    with open(path, 'rb') as datafile:
        data = pickle.load(datafile)
    return data

def create_folders(*folders):
    """Create folders if they don't exist

        Args:
            folders (unnamed args): folders to create

        Returns:
            None

    """
    for folder in folders:
        if not exists(folder):
            makedirs(folder)
