import numpy as np
import cv2


def scale_crop(rect_to_scale, cropping_bb, original_sizes):
    # Rescale func from sahi coord to original coord
    x0, y0, x1, y1 = rect_to_scale
    original_w, original_h = original_sizes

    min_y = min(y0, y1)
    max_y = max(y0, y1)
    min_x = min(x0, x1)
    max_x = max(x0, x1)

    min_x = min_x + cropping_bb[0]
    min_y = min_y + cropping_bb[1]

    max_x = max_x + (original_w - cropping_bb[2]) + (cropping_bb[0] - (original_w - cropping_bb[2]))
    max_y = max_y + (original_h - cropping_bb[3]) + (cropping_bb[1] - (original_h - cropping_bb[3]))

    return min_x, min_y, max_x, max_y

def fix_coords(tags_coords):
    x1, y1, x2, y2 = tags_coords
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)
    return xmin, ymin, xmax, ymax


def fix_coords_line(line):
    start, end = line[:2], line[2:]
    line = sorted([start, end])
    line = [*line[0], *line[1]]
    return tuple(line)


def scale_crop_point(point, size, bbox):
    original_width, original_height = size
    # Calculate the width and height of the bbox
    rel_x = point[0] / original_width
    rel_y = point[1] / original_height

    # Step 2: Apply cropping transformation to the relative coordinates
    new_x = (rel_x - bbox[0] / original_width) * original_width
    new_y = (rel_y - bbox[1] / original_height) * original_height

    return new_x, new_y


def check_line_type(line):
    if line[1] == line[3]:
        return 'horizontal'
    elif line[0] == line[2]:
        return 'vertical'
    else:
        return 'other'


def line_v_h_intersection(line1, line2, tol=1):
    check_vh_intersect = lambda x, y: (x[0] <= y[0] - tol <= x[2] or x[0] <= y[0] + tol <= x[2]) \
                                      and (y[1] <= x[1] - tol <= y[3] or y[1] <= x[1] + tol <= y[3])
    if line1 == line2:
        return True, [line1[0], line1[1]]

    elif line1[1] == line1[3] and line2[0] == line2[2]:
        return check_vh_intersect(line1, line2), [line2[0], line1[1]]

    elif line1[0] == line1[2] and line2[1] == line2[3]:
        return check_vh_intersect(line2, line1), [line1[0], line2[1]]

    return False, None


def euclidean_dist(point1, point2):
    x0, y0 = point1
    x1, y1 = point2

    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def get_projection_area_coords(point, anchor_point, width, height):
    point_x, point_y = point
    anchor_point_x, anchor_point_y = anchor_point
    x_max = (width - point_x) + anchor_point_x
    y_max = (height - point_y) + anchor_point_y
    x_min, y_min = x_max - width, y_max - height
    return x_min, y_min, x_max, y_max


def point_cw_rotate(coord, rotation, mediabox):
    x, y = coord
    mediabox_width = int(mediabox[2] - mediabox[0])
    mediabox_height = int(mediabox[3] - mediabox[1])

    if rotation == 90:
        return y, mediabox_height - x
    elif rotation == 180:
        return mediabox_width - x, mediabox_height - y
    elif rotation == 270:
        return mediabox_width - y, x
    else:
        return coord


def compute_slope_intercept(line):
    x1, y1, x2, y2 = line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def line_v_h_o_intersection(line1, line2, tol1=1,
                            to_round=True):
    # line1 is assumed to be a vertical/horizontal line
    # line2 is assumed to be an inclined line
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4, m2, b2 = line2

    if x1 == x2:  # line1 is a vertical line
        x_int = x1
        y_int = m2 * x_int + b2
    elif y1 == y2:  # line1 is a horizontal line
        y_int = y1
        x_int = (y_int - b2) / m2

        # check if intersection point is within the bounds of both lines
    if to_round:
        x_int = round(x_int)
        y_int = round(y_int)

    min_x1, max_x1 = min(x1, x2), max(x1, x2)
    min_y1, max_y1 = min(y1, y2), max(y1, y2)

    min_x2, max_x2 = min(x3, x4), max(x3, x4)
    min_y2, max_y2 = min(y3, y4), max(y3, y4)

    if (
            min_x1 - tol1 <= x_int <= max_x1 + tol1
            and min_x2 - tol1 <= x_int <= max_x2 + tol1
            and min_y1 - tol1 <= y_int <= max_y1 + tol1
            and min_y2 - tol1 <= y_int <= max_y2 + tol1
    ):

        return True, (x_int, y_int)

    else:
        return False, None


def line_o_o_intersection(line1, line2, tol1=1):
    # line1 is assumed to be a vertical/horizontal line
    # line2 is assumed to be an inclined line
    x1, y1, x2, y2, m1, b1 = line1
    x3, y3, x4, y4, m2, b2 = line2
    if m1 == m2:
        return False, None
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1

    # check if intersection point is within the bounds of both lines
    x_int = round(x_int)
    y_int = round(y_int)

    min_x1, max_x1 = min(x1, x2), max(x1, x2)
    min_y1, max_y1 = min(y1, y2), max(y1, y2)

    min_x2, max_x2 = min(x3, x4), max(x3, x4)
    min_y2, max_y2 = min(y3, y4), max(y3, y4)

    if (
            min_x1 - tol1 <= x_int <= max_x1 + tol1
            and min_x2 - tol1 <= x_int <= max_x2 + tol1
            and min_y1 - tol1 <= y_int <= max_y1 + tol1
            and min_y2 - tol1 <= y_int <= max_y2 + tol1
    ):

        return True, (x_int, y_int)

    else:
        return False, None


def line_intersection(line1, line2):
    if (line1[0] == line1[2] or line1[1] == line1[3]) and \
            (line2[0] == line2[2] or line2[1] == line2[3]):
        return line_v_h_intersection(line1, line2, 0)

    elif (line1[0] == line1[2] or line1[1] == line1[3]) and \
            not (line2[0] == line2[2] or line2[1] == line2[3]):
        m, b = compute_slope_intercept(line2)
        return line_v_h_o_intersection(line1, [*line2, m, b], 0)

    elif (line2[0] == line2[2] or line2[1] == line2[3]) and \
            not (line1[0] == line1[2] or line1[1] == line1[3]):
        m, b = compute_slope_intercept(line1)
        return line_v_h_o_intersection(line2, [*line1, m, b], 0)

    else:
        m1, b1 = compute_slope_intercept(line1)
        m2, b2 = compute_slope_intercept(line2)
        return line_o_o_intersection([*line1, m1, b1], [*line2, m2, b2], 0)


def scale_crop_point(point, size, bbox):
    if len(size) == 3:
        original_width, original_height, _ = size
    if len(size) == 2:
        original_width, original_height = size
    # Calculate the width and height of the bbox
    rel_x = point[0] / original_width
    rel_y = point[1] / original_height
    # Step 2: Apply cropping transformation to the relative coordinates
    new_x = (rel_x - bbox[0] / original_width) * original_width
    new_y = (rel_y - bbox[1] / original_height) * original_height
    return [int(np.round(new_x)), int(np.round(new_y))]


def scale_bbox_by_Bbox(Bbox, bbox, img_size):
    # left, top, right, bottom
    # x_min, y_max, x_max, y_min
    x_min_B, y_max_B, x_max_B, y_min_B = Bbox
    x_min_b, y_max_b, x_max_b, y_min_b = bbox
    return scale_crop_point((x_min_b, y_max_b), img_size, Bbox) + scale_crop_point((x_max_b, y_min_b), img_size, Bbox)


def get_points_of_intersection_scale(list_of_lines, BBox_segment_page, img_size):
    dct = {}
    for line0 in list_of_lines:
        for line1 in list_of_lines:
            if line0['text'] != line1['text']:
                res = line_intersection(scale_bbox_by_Bbox(BBox_segment_page, line0['grid'], img_size),
                                        scale_bbox_by_Bbox(BBox_segment_page, line1['grid'], img_size))
                if res[1] is not None:
                    dct[(line0['text'], line1['text'])] = res[1]
    return dct


def get_page_rects_intersection_info(w1, h1, w2, h2, H):
    img1_rect = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]],
                         dtype=np.float32)
    img2_rect = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]],
                         dtype=np.float32)

    img2_rect_transformed = cv2.transform(np.array([img2_rect]), H)[0]

    x_left = max(img1_rect[0, 0], img2_rect_transformed[0, 0])
    y_top = max(img1_rect[0, 1], img2_rect_transformed[0, 1])
    x_right = min(img1_rect[2, 0], img2_rect_transformed[2, 0])
    y_bottom = min(img1_rect[2, 1], img2_rect_transformed[2, 1])

    # Compute intersection width and height
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    # Compute areas
    intersection_area = intersection_width * intersection_height
    img1_area = w1 * h1

    # Compute the proportion of the intersection area to the area of the first image
    area_proportion = intersection_area / img1_area
    side_length_proportion = max(intersection_width / w1, intersection_height / h1)

    return area_proportion, side_length_proportion
