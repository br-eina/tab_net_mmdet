"""Provides high-level support for connected components processing"""
import pickle
import cv2
from table_blocks_det import utils

def display_centroids(image, centroids, rad=6, color=(0, 0, 255), width=-1):
    """Display centroids as circles for each connected component.

    Args:
        image (np.array): background image to draw centroids on.
        centroids (np.array): coordinates of centroids.
        rad (int): radius of centroid. Defaults to 6.
        color (tuple, optional): color (B, G, R) of centroids. Defaults to (0, 0, 255): red.
        width (int, optional): width of centroids. Defaults to -1 (fill the circle).

    Returns:
        None

    """
    image_to_show = image.copy()
    for coord in centroids:
        cv2.circle(image_to_show, (int(coord[0]), int(coord[1])), rad, color, width)
    utils.show_image(image_to_show)

def display_bboxes(image, stats, color=(0, 0, 255), width=3):
    """Display bboxes for each connected component.

        Args:
            image (np.array): background image to draw bboxes on.
            stats (list): CC stats converted to list.
            color (tuple, optional): color (B, G, R) of bbox. Defaults to (0, 0, 255): red.
            width (int, optional): width of centroids. Defaults to 3.

        Returns:
            None

    """
    image_to_show = image.copy() # TODO: combine display_bboxes() and save_bboxes
    for stat in stats:
        p_1 = (stat[0], stat[1])
        p_2 = (stat[0] + stat[2], stat[1] + stat[3])
        cv2.rectangle(image_to_show, p_1, p_2, color, width)
    utils.show_image(image_to_show)

def mean_height(stats):
    """Calculate the mean height of connected components stats.

    Args:
        stats (list): CC stats converted to list

    Returns:
        avg_height (int): mean height of CC

    """
    height_sum = 0
    for stat in stats:
        height_sum += stat[3]
    avg_height = height_sum // len(stats)
    return avg_height

def remove_small_objects(stats, height=3, width=6):
    """Removes small connected components.

    Args:
        stats (list): CC stats converted to list
        height (int, optional): height of CC. Defaults to 3.
        width (int, optional): width of CC. Defaults to 6.

    Returns:
        processed_stats (list): stats without small CC

    """
    processed_stats = []
    for stat in stats:
        if not (stat[2] < width and stat[3] < height) or \
               (stat[2] < height and stat[3] < width):
            processed_stats.append(stat)
    return processed_stats

def save_bboxes(image, image_name, stats, cc_type, color=(0, 0, 255), width=3, subfolder=None):
    """Save the image with bboxes around each connected component.

        Args:
            image (np.array): background image to draw bboxes on
            image_name (str): image name
            stats (list): CC stats converted to list.
            cc_type (str): Type of CC (line, text, ...).
                Defines the name of saved image.
            color (tuple, optional): color (B, G, R) of bbox. Defaults to (0, 0, 255): red.
            width (int, optional): width of bbox. Defaults to 3.
            subfolder (str, optional): specified subfolder in table_blocks_det/results/{imagename}.
                Defaults to None.

        Returns:
            None

    """
    image_to_show = image.copy()
    for stat in stats:
        p_1 = (stat[0], stat[1])
        p_2 = (stat[0] + stat[2], stat[1] + stat[3])
        cv2.rectangle(image_to_show, p_1, p_2, color, width)
    if subfolder:
        path = f'table_blocks_det/results/{image_name}/{subfolder}/{image_name}_{cc_type}.jpg'
    else:
        path = f'table_blocks_det/results/{image_name}/{image_name}_{cc_type}.jpg'
    utils.save_image(image_to_show, path)

def dump_stats(image_name, stats, cc_type, folder='table_blocks_det/data'):
    """Save stats in JSON format as .data.

        Args:
            image_name (str): image name
            stats (list): CC stats converted to list.
            cc_type (str): Type of CC (line, text, ...).
                Defines the name of saved data.
            folder (str, optional): specified folder.
                Defaults to 'table_blocks_det/data'.

        Returns:
            None

    """
    data = [] # TODO: mb construct class of symbol
    for stat in stats:
        data.append({'x': stat[0],
                     'y': stat[1],
                     'w': stat[2],
                     'h': stat[3],
                     'A': stat[4]})
    filename = f'{folder}/{cc_type}_{image_name}.data'
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def detect_text_elements(stats, avg_height, prop_wh=10,
                         minh=2, prop_maxh=4, mina=8, maxa=1500):
    """Detect text elements within all connected components.

        Args:
            stats (list): CC stats converted to list
            avg_height (int): mean heigh of CC
            prop_wh (int, optional): if (width > prop_wh * height) -> it is not text.
                Defaults to 10.
            minh (int, optional): Minimum height of text. Defaults to 2.
            prop_maxh (float, optional): if (height > average_height * prop_maxh) -> it is not text.
                Defaults to 4.
            mina (int, optional): Minimum area of CC. Defaults to 8.
            maxa (int, optional): Maximum area of CC. Defaults to 1500.

        Returns:
            text_stats (list): text CC stats

    """
    text_stats = []
    for stat in stats:
        if (stat[2] < prop_wh * stat[3]) and \
           (minh <= stat[3] <= avg_height * prop_maxh) and \
           (mina <= stat[4] <= maxa):

            text_stats.append(stat)
    return text_stats


def highlight_lines(mask, struct_el):
    """Highlight lines on the mask

        Args:
            mask (np.array): binary mask
            struct_el (np.array): structuring element for forms highlighting

        Returns:
            lines_mask (np.array): lines of specified form

    """
    lines_mask = cv2.erode(mask, struct_el) # erosion: object size decreases
    lines_mask = cv2.dilate(lines_mask, struct_el) # dilation: object size increases
    return lines_mask