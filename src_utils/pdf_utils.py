import fitz
import numpy as np
from collections import Counter
from src_utils.geometry_utils import line_v_h_intersection, fix_coords_line, scale_crop_point, point_cw_rotate
import PIL.Image as pil_image
import io
from src_logging.log_config import setup_logger
logger = setup_logger(__name__)
def fix_cropbox(cropbox,
                transformation_matrix=None):
    # cropbox should be fixed in cases when there is
    # a discrepancy between rectangle and mediabox
    if transformation_matrix:
        return [*list(fitz.Point(cropbox[:2]) * ~transformation_matrix),
                *list(fitz.Point(cropbox[2:]) * ~transformation_matrix)]
    else:
        return cropbox

def insert_blank_image(page, position,
                       width, height):
    blank_img = pil_image.fromarray(
        np.full(shape=(height,
                       width, 3),
                fill_value=255,
                dtype='uint8'))

    bio = io.BytesIO()
    blank_img.save(bio, "JPEG")
    page.insert_image(fitz.Rect(position),
                                        alpha=0,
                                        stream=bio)

def crop_page(doc, page, crop_area, original_w, original_h,
              apply_redactions=True):
    # calculate rectangles for redactions
    if apply_redactions and min(crop_area)>=0:
        rectangles_for_redactions = []
        if crop_area[0] > 0:
            rectangles_for_redactions.append(fitz.Rect(0, 0, crop_area[0], original_h))
        if crop_area[1] > 0:
            rectangles_for_redactions.append(fitz.Rect(0, 0, original_w, crop_area[1]))
        if crop_area[2] < original_w:
            rectangles_for_redactions.append(fitz.Rect(crop_area[2], 0, original_w, original_h))
        if crop_area[3] < original_h:
            rectangles_for_redactions.append(fitz.Rect(0, crop_area[3], original_w, original_h))
        for i in rectangles_for_redactions:
            page.add_redact_annot(i)
        # apply redactions
        page.apply_redactions()
    # set cropbox
    try:
        page.set_cropbox(crop_area)
    except ValueError:
        doc.xref_set_key(page.xref, "CropBox", \
                         f"[{crop_area[0]} {crop_area[1]} {crop_area[2]} {crop_area[3]}]")



def create_empty_doc_w_page(width, height):
    doc = fitz.open()
    _ = doc.new_page(width=width, height=height)
    return doc


def get_crop_area(grids, width, height, to_add_perc=0):
    xs = []
    ys = []
    for i in grids:
        xs.extend([i['xCenter'], i['xCenter'] - i['xRadius'], i['xCenter'] + i['xRadius']])
        ys.extend([i['yCenter'], i['yCenter'] - i['yRadius'], i['yCenter'] + i['yRadius']])
        xs.extend([i['grid'][0], i['grid'][2]])
        ys.extend([i['grid'][1], i['grid'][3]])

    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    if to_add_perc > 0:
        distance_w = max_x - min_x
        distance_h = max_y - min_y
        to_add_x = int(np.round(distance_w * to_add_perc))
        to_add_y = int(np.round(distance_h * to_add_perc))
        max_x += to_add_x
        min_y -= to_add_x
        max_y += to_add_y
        min_x -= to_add_y
    return [min(max(min_x, 0), width), min(max(min_y, 0), height), min(max_x, width), min(max_y, height)]


def scale_grid_lines(grids, scale_factor):
    for i in grids:
        i['xCenter']=int(np.round(i['xCenter']*scale_factor))
        i['yCenter']=int(np.round(i['yCenter']*scale_factor))
        i['grid']= [int(np.round(j*scale_factor)) for j in i['grid']]
    return grids

def scale_crop_grid_lines(grids, size, bbox):
    for i in grids:
        i['xCenter'], i['yCenter'] = list(map(lambda x: int(np.round(x)),scale_crop_point((i['xCenter'], i['yCenter']),\
                                                                   size,bbox)))
        i['grid']= list(map(lambda x: int(np.round(x)),scale_crop_point(i['grid'][:2],\
                                                                   size,bbox)))+\
        list(map(lambda x: int(np.round(x)),scale_crop_point(i['grid'][2:],\
                                                                   size,bbox)))
    return grids

def get_grids_intersection_points(grids_h, grids_v):
    grids_intersections = {}
    for grid_h in grids_h:
        for grid_v in grids_v:
            flag, point = line_v_h_intersection(fix_coords_line(grid_h['grid']),
                                    fix_coords_line(grid_v['grid']))
            if flag:
                grids_intersections[(grid_h['text'], grid_v['text'])] = point
    return grids_intersections

def get_grids_intersection_points_overall(intersection_points, overall_grids):
    intersection_points_list = list(intersection_points.keys())
    intersection_points_overall = {}
    for pair in intersection_points_list:
        grid1, grid2 = [i for i in overall_grids if i['text']==pair[0]][0],\
                             [i for i in overall_grids if i['text']==pair[1]][0]
        flag, point = line_v_h_intersection(fix_coords_line(grid1['grid']),
                                        fix_coords_line(grid2['grid']))
        if flag:
            intersection_points_overall[pair] = point
    return intersection_points_overall

def select_scale(scale):
    counter = Counter(scale)
    max_occurencies = max(counter.items(), key=lambda x: x[1])[0]
    if Counter(list(counter.values()))[max_occurencies]>1:
        return np.median(scale)
    else:
        return max(counter.items(), key=lambda x: x[1])[0]

def rotate_grids_data(grids, rotation, mediabox):
    if rotation!=0:
        for i in grids:
            center_coord = i['xCenter'], i['yCenter']
            i['xCenter'], i['yCenter'] = point_cw_rotate(center_coord, rotation, mediabox)
            i['grid'] = [*point_cw_rotate(i['grid'][:2], rotation, mediabox),\
            *point_cw_rotate(i['grid'][2:], rotation, mediabox)]
            if rotation==90 or rotation==270:
                i['xRadius'], i['yRadius'] = i['yRadius'], i['xRadius']
    return grids
