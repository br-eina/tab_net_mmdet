"""Binarizing and detection of text and lines"""
from copy import deepcopy as copy
import pickle
import numpy as np
import cv2
from table_blocks_det import utils, cc_utils

def make_lines_mask(labels, lines_indexes, thresh):
    """Make 'lines' mask from labels of connected components.

        Args:
            labels (np.array): labels of CC
            lines_indexes (list): indexes of 'lines' in labels
            thresh (np.array): thresholded image

        Returns:
            lines_mask (np.array): 'lines' mask

    """
    lines_mask = thresh[:] # TODO: numpy.shape TODO: more general function
    lines_mask = 0
    for label in range(1, len(labels)):
        if label in lines_indexes:
            mask = np.array(labels, dtype=np.uint8)
            mask[labels == label] = 255
            mask[labels != label] = 0
            lines_mask = lines_mask + mask
    if not lines_mask.any():
        lines_mask = thresh[:]
        lines_mask[0:thresh.shape[0], 0:thresh.shape[1]] = 0
    return lines_mask

def detect_lines(stats, avg_height, prop_wh=30, prop_maxh=4):
    """Detect 'lines' - objects that are definetily differ from text.

    If either (width < prop_wh * heigh) or (heigh < prop_maxh * avg_height):
    The object is a 'line'.

        Args:
            stats (list): CC stats converted to list
            avg_height (int): mean height of CC
            prop_wh (int, optional): refer to the description. Defaults to 30.
            prop_maxh (int, optional): refer to the description. Defaults to 4.

        Returns:
            stats_lines (list): CC 'line' stats
            line_indexes (list): indexes of lines in original stats

    """
    stats_lines, lines_indexes = [], []
    for ind, stat in enumerate(stats):
        if (stat[2] < prop_wh * stat[3]) ^ (stat[3] < prop_maxh * avg_height):
            stats_lines.append(stat)
            lines_indexes.append(ind + 1)
    return stats_lines, lines_indexes

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

def mask_to_line_stats(mask, avg_height):
    """Extract line_stats from mask

        Args:
            mask (np.array): mask from which line_stats are extracted
            avg_heigh (int): mean height of CC

        Returns:
            line_stats (list): CC line stats converted to list

    """
    _, _, stats_np, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats_np[1:].tolist()
    line_stats, _ = detect_lines(stats, avg_height)
    return line_stats

def main(image_name, image_path):
    """Main script for detecting text and lines"""
    image = cv2.imread(image_path)

    # Thresholding:
    thresh = utils.threshold_image(image)
    thresh = 255 - thresh
    binary_folder = f'table_blocks_det/results/{image_name}/binary'
    utils.create_folders(binary_folder)
    utils.save_image(thresh, path=f'{binary_folder}/{image_name}_binary.jpg')

    # Get connected components for whole binary image:
    _, labels, stats_np, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    stats = stats_np[1:].tolist() # TODO: numpy array

    # Get stats for text elements:
    avg_height = cc_utils.mean_height(stats)
    text_el_stats = cc_utils.detect_text_elements(stats, avg_height)
    cc_utils.save_bboxes(image, image_name, text_el_stats, # Debug text bboxes saving
                         cc_type='text_1', subfolder='text')

    # Get stats for 'lines':
    stats_lines, lines_indexes = detect_lines(stats, avg_height)
    cc_utils.save_bboxes(image, image_name, stats_lines, # Debug 'lines' saving
                         cc_type='detect_lines', subfolder='lines')

    # Save 'lines' mask:
    lines_mask = make_lines_mask(labels, lines_indexes, thresh) # TODO: fix retval
    utils.save_image(lines_mask, f'{binary_folder}/{image_name}_lines_full.jpg')

    # Isolate horizontal and vertical lines using morphological operations:
    scale = 50
    horizontal_mask, vertical_mask = copy(lines_mask), copy(lines_mask)

    horizontal_size = int(horizontal_mask.shape[1] / scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = highlight_lines(horizontal_mask, horizontal_structure)

    vertical_size = int(vertical_mask.shape[0] / scale) # TODO: mb diff scale for vert?
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = highlight_lines(vertical_mask, vertical_structure)

    # Get lines from morphological operations:
    morph_lines_mask = horizontal_mask + vertical_mask
    kernel = np.ones((3, 3), np.uint8)
    dilated_lines = cv2.dilate(morph_lines_mask, kernel, iterations=1)
    utils.save_image(dilated_lines, # Debug dilated_lines saving
                     f'{binary_folder}/{image_name}_lines_dilated.jpg')

    # Get detached elements from lines and tables:
    detached_elem_mask = lines_mask - dilated_lines
    detached_elem_path = f'{binary_folder}/{image_name}_detached_elem.jpg'
    utils.save_image(detached_elem_mask, detached_elem_path)

    # Read detached elem image and threshold it in a simplier way:
    elem_image = cv2.imread(detached_elem_path) # TODO: avoid image reading
    elem_gray_image = utils.grayscale_image(elem_image)
    _, elem_thresh = cv2.threshold(elem_gray_image, 127, 255, cv2.THRESH_BINARY)

    # Connected components for detached text elements:
    _, _, stats_np_det, _ = cv2.connectedComponentsWithStats(elem_thresh, connectivity=8)
    stats_det = stats_np_det[1:].tolist()

    # Detect text from detached elements (might be lines, stamps, etc.):
    remaining_text_el_stats = cc_utils.detect_text_elements(stats_det, avg_height)
    cc_utils.save_bboxes(image, image_name, remaining_text_el_stats,
                         cc_type='text_2', subfolder='text')

    # Combine all text elements:
    all_text_stats = text_el_stats + remaining_text_el_stats

    # Connected components for lines:
    dil_lines_stats = mask_to_line_stats(dilated_lines, avg_height)
    vert_lines_stats = mask_to_line_stats(vertical_mask, avg_height)
    hor_lines_stats = mask_to_line_stats(horizontal_mask, avg_height)

    # Save all the data:
    cc_type_dict = {'conn_comp': stats,
                    'text': all_text_stats,
                    'lines': stats_lines,
                    'lines_vert': vert_lines_stats,
                    'lines_hor': hor_lines_stats,
                    'lines_dil': dil_lines_stats}
    for key in cc_type_dict:
        cc_utils.dump_stats(image_name, cc_type_dict[key],
                            cc_type=key)
        cc_utils.save_bboxes(image, image_name,
                             cc_type_dict[key], cc_type=key)
    with open(f"table_blocks_det/data/avg_h_{image_name}.data", 'wb') as outfile:
        # store the data as binary data stream
        pickle.dump(avg_height, outfile)

if __name__ == "__main__":
    _IMAGE_NAME = "image_test"
    _IMAGE_PATH = "{}.jpg".format(_IMAGE_NAME)
    main(_IMAGE_NAME, _IMAGE_PATH)
    print(__name__)
