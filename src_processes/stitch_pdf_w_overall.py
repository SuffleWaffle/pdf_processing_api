from collections import Counter
from itertools import chain

from src_utils.geometry_utils import check_line_type, get_projection_area_coords
from src_utils.pdf_utils import get_crop_area, crop_page, scale_crop_grid_lines, \
    create_empty_doc_w_page, scale_grid_lines, get_grids_intersection_points, get_grids_intersection_points_overall, \
    select_scale, rotate_grids_data
import fitz
from src_logging.log_config import setup_logger
import numpy as np
logger = setup_logger(__name__)


def stitch_pdf_w_overall(overall_plan_doc,
                         overall_plan_grids,
                         docs_list,
                         grids_list,
                         config):

    # get rotations
    rotation_set = list(set([i[0].rotation for i in docs_list if i[0].rotation!=0]+\
                            [overall_plan_doc[0].rotation]))
    if rotation_set:
        rotation = rotation_set[0]
    else:
        rotation = 0
    # process overall grids
    overall_plan_grids = [i for i in overall_plan_grids if check_line_type(i['grid']) \
                     in ['vertical', 'horizontal']]

    # get center
    for i in overall_plan_grids:
        i['xCenter'] = (i['bbox'][0]+i['bbox'][2])//2
        i['yCenter'] = (i['bbox'][3]+i['bbox'][1])//2
        i['xRadius'] = abs(i['bbox'][0]-i['bbox'][2])//2
        i['yRadius'] = abs(i['bbox'][3] - i['bbox'][1])//2

    # rotate overall plan and grids
    if overall_plan_doc[0].rotation!=rotation:
        overall_plan_doc[0].set_rotation(rotation)
    else:
        overall_plan_doc[0].set_rotation(0)

    overall_plan_grids = rotate_grids_data(overall_plan_grids,
                                           rotation,
                                           overall_plan_doc[0].rect)

    # get overall cropbox, new width and height
    overall_original_width, overall_original_height = overall_plan_doc[0].rect.br
    overall_cropbox = get_crop_area(overall_plan_grids,
                                    overall_original_width, overall_original_height,
                                    **config['get_crop_area_overall'])
    overall_plan_width = overall_cropbox[2] - overall_cropbox[0]
    overall_plan_height = overall_cropbox[3] - overall_cropbox[1]

    # get scale and other info
    scale = []
    processed_docs = []
    for c, doc in enumerate(docs_list):

        grids = grids_list[c]
        grids = [i for i in grids if check_line_type(i['grid']) \
                 in ['vertical', 'horizontal']]
        #add centers and radius
        for i in grids:
            i['xCenter'] = (i['bbox'][0] + i['bbox'][2]) // 2
            i['yCenter'] = (i['bbox'][3] + i['bbox'][1]) // 2
            i['xRadius'] = abs(i['bbox'][0] - i['bbox'][2]) // 2
            i['yRadius'] = abs(i['bbox'][3] - i['bbox'][1]) // 2

        page = doc[0]
        page_rotation = page.rotation
        if page_rotation!=rotation:
            page.set_rotation(rotation)
        else:
            page.set_rotation(0)

        grids = rotate_grids_data(grids, rotation,
                                  page.rect)

        # get intersecting grids
        intersecting_grids_names = set([i['text'] for i in overall_plan_grids]).intersection([i['text'] for i in grids])
        intersecting_grids = [i for i in grids if i['text'] in intersecting_grids_names]
        # get scale
        intersecting_grids_v = [i for i in intersecting_grids if check_line_type(i['grid']) == 'vertical']
        intersecting_grids_h = [i for i in intersecting_grids if check_line_type(i['grid']) == 'horizontal']

        if len(intersecting_grids_v) >= 2:
            intersecting_grids_overall_v = [j for j in overall_plan_grids
                                            if j['text'] in [i['text'] for i in intersecting_grids_v]]
            close_grids = sorted(intersecting_grids_v, key=lambda x: x['xCenter'])

            close_grids_pairs = [close_grids[i:i + 2] for i in range(0, len(close_grids) - 1, 1)]
            diffs_grids_pos = [abs(pair[0]['grid'][0] - pair[1]['grid'][0])
                               for pair in close_grids_pairs]
            arg_max = np.argmax(diffs_grids_pos)
            diff_grids_pos = diffs_grids_pos[arg_max]
            pair_to_use = close_grids_pairs[arg_max]

            close_grids_overall = [i for i in intersecting_grids_overall_v
                                   if i['text'] in [j['text'] for j in pair_to_use]]

            diff_grids_overall_pos = abs(close_grids_overall[0]['grid'][0] - close_grids_overall[1]['grid'][0])

        else:
            intersecting_grids_overall_h = [j for j in overall_plan_grids
                                            if j['text'] in [i['text'] for i in intersecting_grids_h]]
            close_grids = sorted(intersecting_grids_h, key=lambda x: x['yCenter'])
            close_grids_pairs = [close_grids[i:i + 2] for i in range(0, len(close_grids) - 1, 1)]
            diffs_grids_pos = [abs(pair[0]['grid'][1] - pair[1]['grid'][1])
                               for pair in close_grids_pairs]
            arg_max = np.argmax(diffs_grids_pos)

            diff_grids_pos = diffs_grids_pos[arg_max]
            pair_to_use = close_grids_pairs[arg_max]
            close_grids_overall = [i for i in intersecting_grids_overall_h
                                   if i['text'] in [j['text'] for j in pair_to_use]]

            diff_grids_overall_pos = abs(close_grids_overall[0]['grid'][1] - close_grids_overall[1]['grid'][1])

        scale.append(diff_grids_pos / diff_grids_overall_pos)
        # crop area
        w, h = page.rect.br
        crop_area = get_crop_area(grids, w, h,
                                  **config['get_crop_area'])
        crop_page(doc, page, crop_area, original_w=w, original_h=h)
        intersecting_grids_h = [i for i in intersecting_grids if check_line_type(i['grid']) == 'horizontal']
        intersecting_grids_v = scale_crop_grid_lines(intersecting_grids_v, size=(w, h), bbox=crop_area)
        intersecting_grids_h = scale_crop_grid_lines(intersecting_grids_h, size=(w, h), bbox=crop_area)
        processed_docs.append([doc, intersecting_grids_v, intersecting_grids_h])

    # estimate scale
    scale = [i for i in scale if i > 1]
    scale = select_scale(scale)

    logger.info('Estimated scale, cropped pdfs')

    # create empty doc for stitching
    empty_overall_doc = create_empty_doc_w_page(width=int(np.round(scale*overall_plan_width)),
                                                height=int(np.round(scale*overall_plan_height)))

    # scale + scale-crop overall lines
    overall_plan_grids = scale_crop_grid_lines(overall_plan_grids,
                                               size=(overall_original_width, overall_original_height),
                                               bbox=overall_cropbox)
    overall_plan_grids = scale_grid_lines(overall_plan_grids, scale)

    # stitch
    empty_overall_doc_page = empty_overall_doc[0]
    intersections_list = []
    for ent in processed_docs:
        # get intersection points
        doc, intersecting_grids_v, intersecting_grids_h = ent
        intersection_points = get_grids_intersection_points(intersecting_grids_h, intersecting_grids_v)
        intersection_points_overall = get_grids_intersection_points_overall(intersection_points, overall_plan_grids)
        intersections_list.append([intersection_points, intersection_points_overall])

    num_docs = len(processed_docs)
    pairs_counter = Counter(list(chain(*[list(i[0].keys()) for i in intersections_list])))
    match_pairs = [k for k, v in pairs_counter.items() if v == num_docs]
    if not match_pairs:
        max_val = max(pairs_counter.values())
        match_pairs = [k for k, v in pairs_counter.items() if v == max_val]

    for c, ent in enumerate(processed_docs):
        doc = ent[0]
        intersection_points, intersection_points_overall = intersections_list[c]
        to_use = set(intersection_points_overall.keys()).intersection(intersection_points.keys())
        intersected_pairs = set(to_use).intersection(match_pairs)
        if not intersected_pairs:
            intersected_pairs = to_use

        already_there = set()
        page_rects = []
        for pair in intersected_pairs:
            if not already_there.intersection(pair):
                stitch_point = intersection_points[pair]
                anchor_point = intersection_points_overall[pair]
                doc_w, doc_h = doc[0].rect.br
                x_min, y_min, x_max, y_max = get_projection_area_coords(stitch_point, anchor_point,
                                                                        width=doc_w, height=doc_h)
                page_rects.append((x_min, y_min, x_max, y_max))
                already_there.update(pair)

        counter_page_rects = Counter(page_rects)
        max_frequency = max(counter_page_rects.values())
        if Counter(list(counter_page_rects.values()))[max_frequency] > 1:
            page_rect = fitz.Rect(*list(np.mean(np.array(page_rects), axis=0)))
        else:
            page_rect = fitz.Rect(*max(Counter(page_rects).items(), key=lambda x: x[1])[0])
        empty_overall_doc_page.show_pdf_page(page_rect, doc, 0)
    empty_overall_doc_page.set_rotation(rotation)
    logger.info('Stitched PDFs')

    return empty_overall_doc.write()
