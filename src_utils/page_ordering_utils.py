import enum
from longest_increasing_subsequence import longest_increasing_subsequence, longest_decreasing_subsequence

import numpy as np
import re

class GridNameRelation(enum.Enum):
    LESS_THAN_ALL = -3
    IN_BETWEEN = -2
    GREATER_THAN_ALL = -1
    NOT_SET = -10


class RelativePlacementTypes(enum.Enum):
    OVERLAP = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4

def fill_list_1_vs_list_2_relation(list_1, list_2, list_1_result, list_2_result):
    set_2_min = list_2[0][0]
    set_2_max = list_2[-1][0]
    for i, (name, idx, _) in enumerate(list_1):
        if name < set_2_min:
            list_1_result[i] = GridNameRelation.LESS_THAN_ALL.value
        elif name > set_2_max:
            list_1_result[i] = GridNameRelation.GREATER_THAN_ALL.value
        else:
            for j, (name_2, idx_2, _) in enumerate(list_2):
                if name == name_2:
                    list_1_result[i] = j
                    list_2_result[j] = i
                    break
                elif name < name_2:
                    list_1_result[i] = GridNameRelation.IN_BETWEEN.value
                    break


def get_transformed_grid_lines(original_grid_lines, page_region, *, keep_cropped = False, page_name: str = None) -> list[dict]:
    grid_lines = original_grid_lines[:]
    resulting_grid_lines = []

    x_max = page_region[2]
    y_max = page_region[3]
    for i in range(len(grid_lines)):
        grid_lines[i]['bbox'][0] -= page_region[0]
        grid_lines[i]['bbox'][1] -= page_region[1]
        grid_lines[i]['bbox'][2] -= page_region[0]
        grid_lines[i]['bbox'][3] -= page_region[1]

        grid_lines[i]['grid'][0] -= page_region[0]
        grid_lines[i]['grid'][1] -= page_region[1]
        grid_lines[i]['grid'][2] -= page_region[0]
        grid_lines[i]['grid'][3] -= page_region[1]

        cropped = False

        x1, y1, x2, y2 = grid_lines[i]['grid']
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            if keep_cropped:
                if max(x1, x2) < 0 or max(y1, y2) < 0:
                    cropped = True
                else:
                    # grid_lines[i]['grid'][grid_lines[i]['grid']<0] = 0
                    if x1 < 0:
                        grid_lines[i]['grid'][0] = 0
                    if x2 < 0:
                        grid_lines[i]['grid'][2] = 0
                    if y1 < 0:
                        grid_lines[i]['grid'][1] = 0
                    if y2 < 0:
                        grid_lines[i]['grid'][3] = 0

            else:
                cropped = True

        if not cropped:
            if x1 > x_max or x2 > x_max or y1 > y_max or y2 > y_max:
                # logger.warning("Some coordinates are outside of the page region for gridline %s after cropping on page %s - gridline ignored.",
                #                grid_lines[i]['text'], page_name)
                if keep_cropped:
                    if min(x1, x2) > x_max or min(y1, y2) > y_max:
                        cropped = True
                    else:
                        if x1 > x_max:
                            grid_lines[i]['grid'][0] = x_max
                        if x2 > x_max:
                            grid_lines[i]['grid'][2] = x_max
                        if y1 > y_max:
                            grid_lines[i]['grid'][1] = y_max
                        if y2 > y_max:
                            grid_lines[i]['grid'][3] = y_max

                else:
                    cropped = True

        if not cropped:
            resulting_grid_lines.append(grid_lines[i])

    return resulting_grid_lines

# using horizontal/vertical grid lines index, normalizes names (lowercase, padding for numeric etc.),
# adds 'name' key to the original grid lines dictionary, and also returns
# list of tuples (name, index, coord) for all non-skipped grid line names and dictionary
# which maps original grid line name (from the 'text' field) to index of the grid line in horizontal/vertical grid lines list
def normalize_names(aligned_grids, main_coord_index):  # main_coord_index == 0 (x) for vertical and 1 (y) for horizontal
    grid_names_list = []
    name_to_grid_dict = {}

    # grid can be names as "A", "B", "C", "A.12" or "1", "12", "13", "12.3" etc.
    for i, grid in enumerate(aligned_grids):
        grid_label = grid['text']

        # we need tuple of 3 elements for every grid line - prefix, integer part, decimal part
        # use regex to get prefix consisting of letters, then integer part and then decimal part. Every part is optional
        # if decimal part is present, and integer part is not, then set integer part to 0
        # if integer part is present, and decimal part is not, then set decimal part to 0
        # if there is no textual part, then set prefix to '-' (minus sign)

        l, i, d = re.match(r'([a-zA-Z]*)(\d+)?(\.\d+)?', grid_label).groups()
        if l is None:
            l = '-'
        if i is None:
            i = 0
        if d is None:
            d = 0
        name = (l.lower().strip(), int(i), float(d))
        grid['name'] = name

        if i in name_to_grid_dict:
            pass # duplicated grids are filtered before
            # logger.error("Grid name %s already exists in the name to index grids dictionary!!!", name)
        else:
            name_to_grid_dict[name] = i

    # filter out names where alphabetical order is not aligned with the coordinate order
    inc_sec = longest_increasing_subsequence(aligned_grids, key=lambda x: x['name'])
    dec_sec = longest_decreasing_subsequence(aligned_grids, key=lambda x: x['name'])

    correct_grid_sec = inc_sec if len(inc_sec) > len(dec_sec) else dec_sec

    for i, grid in enumerate(correct_grid_sec):
        name = grid['name']
        grid_names_list.append((name, i, grid['grid'][main_coord_index]))

    grid_names_list.sort(key=lambda x: x[0])

    return grid_names_list, name_to_grid_dict


# split grid lines into horizontal and vertical and other, and sort them
# other grid lines are ignored for now
def prepare_single_page_grid_lines(grid_lines, *, same_coord_delta: int = 2):
    horizontal_lines = []
    vertical_lines = []
    other_lines = []

    for grid_line in grid_lines:
        line = grid_line['grid']
        if abs(line[0] - line[2]) < same_coord_delta:
            vertical_lines.append(grid_line)
        elif abs(line[1] - line[3]) < same_coord_delta:
            horizontal_lines.append(grid_line)
        else:
            other_lines.append(grid_line)

    horizontal_lines.sort(key=lambda x: x['grid'][1])
    vertical_lines.sort(key=lambda x: x['grid'][0])

    return horizontal_lines, vertical_lines, other_lines


def get_grid_names_relations(grid_named_set_1, grid_named_set_2) -> tuple[np.array, np.array]:
    # Grid names are sorted in alphabetical order, and we can use this to find out the relation,
    # because alphabetical order is the same as the order of grid lines along the axis

    # The idea is that we have two parts of the same document, and we want to find out how they are related
    # possible cases are: LEFT-RIGHT, RIGHT-LEFT, TOP-BOTTOM, BOTTOM-TOP, with intersection or without.
    # And also we can have a case when parts are not related at all or have very small intersection (like one corner)
    # Here we compare only one set of grids - like horizontal to horizontal or vertical to vertical
    # So answer will be LEQ (less or equal), GEQ, LE, GE, NOT_RELATED
    # We assume that grid line names are sorted in alphabetical order, and we can use this to find out the relation
    # E.g. if in the first set we have A, B, C, D and in the second set we have D, E, F then we can
    # say that the first set is LESS or EQUAL to the second set, because all names from the first set are less or the same as in the second set
    # Another example: if we have A, B, C and D, E, F, G then we also can say that the first set is LESS to the second set

    # we will return two numpy arrays with the same length as the first and second sets,
    # with the next possible values: -3 means "less than all in the other set", -1 means "greater than all in the other set",
    # -2 means "somewhere in between values from the other set, but not coinciding with any of them",
    # and 0 or positive number means the index of the corresponding grid with the same name in the other set

    set_1_result = np.ones(len(grid_named_set_1), dtype=np.int8) * GridNameRelation.NOT_SET.value
    set_2_result = np.ones(len(grid_named_set_2), dtype=np.int8) * GridNameRelation.NOT_SET.value

    fill_list_1_vs_list_2_relation(grid_named_set_1, grid_named_set_2, set_1_result, set_2_result)
    fill_list_1_vs_list_2_relation(grid_named_set_2, grid_named_set_1, set_2_result, set_1_result)

    return set_1_result, set_2_result

# Example relative_orders information for 2 pages one on top of the other (P1_1):
# Here vertical grid lines are ordered from Z to A along x axis - so it's opposite to the natural order
# H1: np.array([-3 - 3 - 3 - 3 - 3 - 3 - 3  0])
# H2: np.array([7 - 1 - 1 - 1 - 1 - 1 - 1 - 1])
# V1: np.array([0  1 - 2  2 - 2  3 - 2  4  5  6  7  8  9 10 11])
# V2: np.array([0  1  3  5  7  8  9 10 11 12 13 14])

# Gets lexicographic relative position for every H/V grid line for the page pair. -3 - grid line name is less than all the name of the other page
# -1 - grid line name is greater than all the names of the other page
# 0 or greater - grid line name exists on both pages, and the value is the index of the corresponding grid line in the other page
# -2 means none of the above is true, usually means some auxiliary grid line, named like "C.41" etc.
# Also gets aggregated stats for the relative position of the grid lines for the page pair - how many grid lines are less, greater, intersect, or other
def get_grid_headers_relative_order(grid_sets, *, page_names: list[str] | None = None):
    relative_orders = []
    aggregated_relative_orders = []
    processed_pages = set()

    i_1 = 0
    for page_idx_1, h_grids_named_1, v_grids_named_1, *_ in grid_sets:
        i_2 = i_1 + 1
        for page_idx_2, h_grids_named_2, v_grids_named_2, *_ in grid_sets[i_1 + 1:]:
            # actually i1 and i2 are the same as page_idx_1 and page_idx_2

            if len(h_grids_named_1) == 0 or len(h_grids_named_2) == 0:
                i_2 += 1
                continue
            elif len(v_grids_named_1) == 0 or len(v_grids_named_2) == 0:
                i_2 += 1
                continue

            h_rel_1, h_rel_2 = get_grid_names_relations(h_grids_named_1, h_grids_named_2)
            v_rel_1, v_rel_2 = get_grid_names_relations(v_grids_named_1, v_grids_named_2)
            relative_orders.append((page_idx_1, page_idx_2, h_rel_1, v_rel_1, h_rel_2, v_rel_2))

            page_idx_1, h_grids_named_1, v_grids_named_1, *_ = grid_sets[i_1]
            page_idx_2, h_grids_named_2, v_grids_named_2, *_ = grid_sets[i_2]

            # calculate relative stats between two pages
            # using the second page relative position stats - so, our result should be read as
            # "how the second page is positioned relative to the first page"
            h_names_less_count = np.sum(h_rel_2 == GridNameRelation.LESS_THAN_ALL.value)
            h_names_greater_count = np.sum(h_rel_2 == GridNameRelation.GREATER_THAN_ALL.value)
            h_names_mess_count = np.sum(h_rel_2 == GridNameRelation.IN_BETWEEN.value)
            h_names_intersec_count = np.sum(h_rel_2 >= 0)

            v_names_less_count = np.sum(v_rel_2 == GridNameRelation.LESS_THAN_ALL.value)
            v_names_greater_count = np.sum(v_rel_2 == GridNameRelation.GREATER_THAN_ALL.value)
            v_names_mess_count = np.sum(v_rel_2 == GridNameRelation.IN_BETWEEN.value)
            v_names_intersec_count = np.sum(v_rel_2 >= 0)

            # if h_names_intersec_count > 0 and v_names_intersec_count > 0:
            #     logger.info("Pages %s and %s HAVE grid intersection", page_names[i_1], page_names[i_2])
            # else:
            #     logger.info("NO grid intersection for %s and %s", page_names[i_1], page_names[i_2])

            # grid_sets contains triple (page_idx, h_grids_named, v_grids_named) for each page
            # and for each grid line we have a tuple:
            # name (transformed for lexicografical sort), index (in origianl aligned grid lines list), coord (x for vertical, y for horizontal))

            # is horizontal grid lines have reversed lexicographical names order in relation to y coordinate, which goes from top (0) to bottom (page height)
            is_horizontal_order_reversed = grid_sets[i_2][1][0][2] > grid_sets[i_2][1][1][2] if len(
                grid_sets[i_2][1]) > 1 else False

            # is vertical grid lines have reversed lexicographical names order in relation to x coordinate which goes from left (0) to right (page width)
            is_vertical_order_reversed = grid_sets[i_2][2][0][2] > grid_sets[i_2][2][1][2] if len(
                grid_sets[i_2][2]) > 1 else False

            aggregated_relative_orders.append((page_idx_1, page_idx_2, {
                'horizontal_less': h_names_less_count,
                'horizontal_greater': h_names_greater_count,
                'horizontal_intersect': h_names_intersec_count,
                'horizontal_mess': h_names_mess_count,
                'horizontal_reversed_koef': -1 if is_horizontal_order_reversed else 1,
                'vertical_less': v_names_less_count,
                'vertical_greater': v_names_greater_count,
                'vertical_intersect': v_names_intersec_count,
                'vertical_mess': v_names_mess_count,
                'vertical_reversed_koef': -1 if is_vertical_order_reversed else 1
            }))

            processed_pages.add(page_idx_1)
            processed_pages.add(page_idx_2)

            i_2 += 1
        i_1 += 1

    return relative_orders, aggregated_relative_orders, processed_pages


# h_grid_name_indices, v_grid_name_indices, h_grid_step, v_grid_step
def prepare_overall_grids_data(grid_sets):
    h_grid_names_set = set()
    v_grid_names_set = set()

    h_grid_steps = []
    v_grid_steps = []

    for page_idx, h_grids_named, v_grids_named, h_grid_name_to_idx, v_grid_name_to_idx in grid_sets:
        # h_grids_named - list of tuples (name, index, coord) for all non-skipped horizontal grid lines
        h_grid_names_set.update([g[0] for g in h_grids_named])
        v_grid_names_set.update([g[0] for g in v_grids_named])

        for i in range(len(h_grids_named) - 1):
            h_grid_steps.append(abs(h_grids_named[i + 1][2] - h_grids_named[i][2]))

        for i in range(len(v_grids_named) - 1):
            v_grid_steps.append(abs(v_grids_named[i + 1][2] - v_grids_named[i][2]))

    # find median step for each axis
    h_grid_steps = np.array(h_grid_steps)
    if h_grid_steps.size > 0:
        h_grid_step = np.median(h_grid_steps)
    else:
        h_grid_step = -1

    v_grid_steps = np.array(v_grid_steps)
    if v_grid_steps.size > 0:
        v_grid_step = np.median(v_grid_steps)
    else:
        v_grid_step = -1

    # convert h_grid_names_set and v_grid_names_set to lists and sort them
    h_grid_names_list = list(h_grid_names_set)
    h_grid_names_list.sort()
    v_grid_names_list = list(v_grid_names_set)
    v_grid_names_list.sort()

    h_grid_name_indices = {name: i for i, name in enumerate(h_grid_names_list)}
    v_grid_name_indices = {name: i for i, name in enumerate(v_grid_names_list)}

    return h_grid_name_indices, v_grid_name_indices, h_grid_step, v_grid_step

def get_relative_placement(placement_axis, horizontal_reversed_koef, vertical_reversed_koef, horizontal_less,
                           horizontal_greater, vertical_less, vertical_greater, *, confidence_threshold: float = 0.7):
    relative_placement = None
    if placement_axis == 0:
        if vertical_less * confidence_threshold > vertical_greater:
            relative_placement = RelativePlacementTypes.LEFT.value if vertical_reversed_koef > 0 else RelativePlacementTypes.RIGHT.value
        elif vertical_greater * confidence_threshold > vertical_less:
            relative_placement = RelativePlacementTypes.RIGHT.value if vertical_reversed_koef > 0 else RelativePlacementTypes.LEFT.value
    elif placement_axis == 1:
        if horizontal_less * confidence_threshold > horizontal_greater:
            relative_placement = RelativePlacementTypes.TOP.value if horizontal_reversed_koef > 0 else RelativePlacementTypes.BOTTOM.value
        elif horizontal_greater * confidence_threshold > horizontal_less:
            relative_placement = RelativePlacementTypes.BOTTOM.value if horizontal_reversed_koef > 0 else RelativePlacementTypes.TOP.value
    elif placement_axis == 2:
        relative_placement = RelativePlacementTypes.OVERLAP.value

    return relative_placement


def is_acceptable_relative_placement(relative_placement, i1, i2, page_dimensions, grid_sets,
                                     horizontal_intersect, h_grid_step, horizontal_reversed_koef, h_grid_name_indices,
                                     vertical_intersect, v_grid_step, vertical_reversed_koef, v_grid_name_indices, page_names):
    is_acceptable = True

    # refactor? copylot copypaste, but works correctly
    if (
            relative_placement == RelativePlacementTypes.TOP.value or relative_placement == RelativePlacementTypes.BOTTOM.value) and horizontal_intersect <= 0:
        # means page 2 assumed to be on the top or bottom from page 1, but no grid lines intersect
        # so we need to check if it's not too far actually, because we can have a case when we have some page in the middle.

        if relative_placement == RelativePlacementTypes.TOP.value:
            # page_1_horizontal_grid_lines = aligned_grid_lines[i1][1]
            # page_2_horizontal_grid_lines = aligned_grid_lines[i2][1]

            if h_grid_step < 0:
                is_acceptable = False

            if is_acceptable:
                # get closest grid lines for both pages, result is a tuple of name, index, coord
                page1_topmost_grid = grid_sets[i1][1][0 if horizontal_reversed_koef > 0 else -1]
                page2_bottommost_grid = grid_sets[i2][1][-1 if horizontal_reversed_koef > 0 else 0]

                name1 = page1_topmost_grid[0]
                name2 = page2_bottommost_grid[0]
                name1_idx = h_grid_name_indices[name1]
                name2_idx = h_grid_name_indices[name2]
                name_distance = abs(name1_idx - name2_idx)

                # Check - simply use number of skipped gridlines as a distance?
                # expected distance between grid lines
                estimated_distance = int(name_distance * h_grid_step)

                # how many pixels to the bottom from the bottommost grid line from the first page we have
                page_1_stripe_height = page1_topmost_grid[2]  # TODO: count for crop region!
                page_2_stripe_height = page_dimensions[i2][1] - page2_bottommost_grid[2]

                if page_1_stripe_height + page_2_stripe_height < estimated_distance:
                    # logger.warning("Page %s is too far from page %s. Estimated distance: %s, possible distance: %s",
                    #                page_names[i2], page_names[i1], estimated_distance,
                    #                page_1_stripe_height + page_2_stripe_height)
                    is_acceptable = False

        elif relative_placement == RelativePlacementTypes.BOTTOM.value:
            # page_1_horizontal_grid_lines = aligned_grid_lines[i1][1]
            # page_2_horizontal_grid_lines = aligned_grid_lines[i2][1]

            if h_grid_step < 0:
                is_acceptable = False

            if is_acceptable:

                # get closest grid lines for both pages, result is a tuple of name, index, coord
                page1_bottommost_grid = grid_sets[i1][1][-1 if horizontal_reversed_koef > 0 else 0]
                page2_topmost_grid = grid_sets[i2][1][0 if horizontal_reversed_koef > 0 else -1]

                name1 = page1_bottommost_grid[0]
                name2 = page2_topmost_grid[0]
                name1_idx = h_grid_name_indices[name1]
                name2_idx = h_grid_name_indices[name2]
                name_distance = abs(name1_idx - name2_idx)

                # expected distance between grid lines
                estimated_distance = int(name_distance * h_grid_step)

                # how many pixels to the top from the topmost grid line from the first page we have
                page_1_stripe_height = page1_bottommost_grid[2]
                page_2_stripe_height = page_dimensions[i2][1] - page2_topmost_grid[2]

                if page_1_stripe_height + page_2_stripe_height < estimated_distance:
                    # logger.warning("Page %s is too far from page %s. Estimated distance: %s, possible distance: %s",
                    #                page_names[i2], page_names[i1], estimated_distance,
                    #                page_1_stripe_height + page_2_stripe_height)
                    is_acceptable = False
    elif (
            relative_placement == RelativePlacementTypes.LEFT.value or relative_placement == RelativePlacementTypes.RIGHT.value) and vertical_intersect <= 0:
        # means page 2 assumed to be on the left or right from page 1, but no grid lines intersect
        # so we need to check if it's not too far actually, because we can have a case when we have some page in the middle.

        if relative_placement == RelativePlacementTypes.LEFT.value:
            # page_1_horizontal_grid_lines = aligned_grid_lines[i1][1]
            # page_2_horizontal_grid_lines = aligned_grid_lines[i2][1]

            if v_grid_step < 0:
                is_acceptable = False

            if is_acceptable:
                # get closest grid lines for both pages, result is a tuple of name, index, coord
                page1_leftmost_grid = grid_sets[i1][2][0 if vertical_reversed_koef > 0 else -1]
                page2_rightmost_grid = grid_sets[i2][2][-1 if vertical_reversed_koef > 0 else 0]

                name1 = page1_leftmost_grid[0]
                name2 = page2_rightmost_grid[0]
                name1_idx = v_grid_name_indices[name1]
                name2_idx = v_grid_name_indices[name2]
                name_distance = abs(name1_idx - name2_idx)

                # expected distance between grid lines
                estimated_distance = int(name_distance * v_grid_step)

                # how many pixels to the right from the rightmost grid line from the first page we have
                page_1_stripe_width = page1_leftmost_grid[2]
                page_2_stripe_width = page_dimensions[i2][0] - page2_rightmost_grid[2]

                if page_1_stripe_width + page_2_stripe_width < estimated_distance:
                    # logger.warning("Page %s is too far from page %s. Estimated distance: %s, possible distance: %s",
                    #                page_names[i2], page_names[i1], estimated_distance,
                    #                page_1_stripe_width + page_2_stripe_width)
                    is_acceptable = False

        elif relative_placement == RelativePlacementTypes.RIGHT.value:
            # page_1_horizontal_grid_lines = aligned_grid_lines[i1][1]
            # page_2_horizontal_grid_lines = aligned_grid_lines[i2][1]

            if v_grid_step < 0:
                is_acceptable = False

            if is_acceptable:
                # get closest grid lines for both pages, result is a tuple of name, index, coord
                page1_rightmost_grid = grid_sets[i1][2][-1 if vertical_reversed_koef > 0 else 0]
                page2_leftmost_grid = grid_sets[i2][2][0 if vertical_reversed_koef > 0 else -1]

                name1 = page1_rightmost_grid[0]
                name2 = page2_leftmost_grid[0]
                name1_idx = v_grid_name_indices[name1]
                name2_idx = v_grid_name_indices[name2]
                name_distance = abs(name1_idx - name2_idx)

                # expected distance between grid lines
                estimated_distance = int(name_distance * v_grid_step)

                # how many pixels to the left from the leftmost grid line from the first page we have
                page_1_stripe_width = page1_rightmost_grid[2]
                page_2_stripe_width = page_dimensions[i2][0] - page2_leftmost_grid[2]

                if page_1_stripe_width + page_2_stripe_width < estimated_distance:
                    # logger.warning("Page %s is too far from page %s. Estimated distance: %s, possible distance: %s",
                    #                page_names[i2], page_names[i1], estimated_distance,
                    #                page_1_stripe_width + page_2_stripe_width)
                    is_acceptable = False

    return is_acceptable

def calculate_pages_relative_placement(page_dimensions, aligned_grid_lines, aggregated_relative_orders, relative_orders,
                                       grid_sets,
                                       h_grid_name_indices, v_grid_name_indices, h_grid_step, v_grid_step,
                                       *,
                                       confidence_threshold: float = 0.7, page_names: list[str] | None = None):
    if page_names is None:
        page_names = [str(i) for i in range(len(aggregated_relative_orders))]

    pages_relative_placement = []
    for i1, i2, relation_data in aggregated_relative_orders:
        horizontal_less = relation_data['horizontal_less']
        horizontal_greater = relation_data['horizontal_greater']
        horizontal_intersect = relation_data['horizontal_intersect']
        horizontal_mess = relation_data['horizontal_mess']
        horizontal_reversed_koef = relation_data['horizontal_reversed_koef']
        vertical_less = relation_data['vertical_less']
        vertical_greater = relation_data['vertical_greater']
        vertical_intersect = relation_data['vertical_intersect']
        vertical_mess = relation_data['vertical_mess']
        vertical_reversed_koef = relation_data['vertical_reversed_koef']

        top_bottom_confidence = abs(horizontal_less - horizontal_greater) - horizontal_intersect
        left_right_confidence = abs(vertical_less - vertical_greater) - vertical_intersect

        use_secondary_placement = False

        if horizontal_intersect > 0 and vertical_intersect > 0 and horizontal_intersect >= max(horizontal_less,
                                                                                               horizontal_greater) and vertical_intersect >= max(
            vertical_less, vertical_greater):
            placement_axis = 2

        elif left_right_confidence > 0 and left_right_confidence > top_bottom_confidence:
            placement_axis = 0
            if top_bottom_confidence > 0 and top_bottom_confidence > 0.75 * left_right_confidence:
                use_secondary_placement = True
        elif top_bottom_confidence > 0 and top_bottom_confidence > left_right_confidence:
            placement_axis = 1
            if left_right_confidence > 0 and left_right_confidence > 0.75 * top_bottom_confidence:
                use_secondary_placement = True
        else:
            continue

        secondary_relative_placement = None
        relative_placement = get_relative_placement(placement_axis, horizontal_reversed_koef, vertical_reversed_koef,
                                                    horizontal_less,
                                                    horizontal_greater, vertical_less, vertical_greater,
                                                    confidence_threshold=confidence_threshold)

        if use_secondary_placement:
            secondary_placement_axis = 1 if placement_axis == 0 else 0
            secondary_relative_placement = get_relative_placement(secondary_placement_axis, horizontal_reversed_koef,
                                                                  vertical_reversed_koef, horizontal_less,
                                                                  horizontal_greater, vertical_less, vertical_greater,
                                                                  confidence_threshold=confidence_threshold)

        if relative_placement is not None and not is_acceptable_relative_placement(relative_placement, i1, i2, page_dimensions, grid_sets,
                                               horizontal_intersect, h_grid_step, horizontal_reversed_koef, h_grid_name_indices,
                                               vertical_intersect, v_grid_step, vertical_reversed_koef, v_grid_name_indices, page_names):
            relative_placement = None

        if secondary_relative_placement is not None and not is_acceptable_relative_placement(secondary_relative_placement, i1, i2, page_dimensions, grid_sets,
                                               horizontal_intersect, h_grid_step, horizontal_reversed_koef, h_grid_name_indices,
                                               vertical_intersect, v_grid_step, vertical_reversed_koef, v_grid_name_indices, page_names):
            secondary_relative_placement = None

        # calculate shifts if we have intersections, or approximate shifts by median step if we don't have intersections
        if vertical_intersect > 0 and horizontal_intersect > 0:
            pass

        if relative_placement is not None or secondary_relative_placement is not None:
            pages_relative_placement.append((i1, i2, (relative_placement, secondary_relative_placement)))

    return pages_relative_placement

def get_pages_relative_placement(page_names, source_grid_lines, page_regions, *, update_grids_by_crop=True):
    if update_grids_by_crop:
        original_grid_lines = []
        for i, page_region in enumerate(page_regions):
            page_name = page_names[i]
            page_grid_lines = source_grid_lines[i] if source_grid_lines is not None else None
            if page_grid_lines is None or len(page_grid_lines) == 0:
                continue

            page_shifted_gridlines = get_transformed_grid_lines(page_grid_lines, page_region, keep_cropped=True, page_name=page_name)
            original_grid_lines.append(page_shifted_gridlines)
    else:
        original_grid_lines = source_grid_lines

    # max_numeric_name = get_max_value_for_numeric_names(original_grid_lines)
    # logger.info("Max numeric grid line name: %s", max_numeric_name)

    # list of tuples, for each page contains a tuple with the lists (horizontal_grids, vertical_grids, other_grids)
    aligned_grid_lines = []
    grid_sets = []
    for idx, (page_name, grid_lines) in enumerate(zip(page_names, original_grid_lines)):
        if grid_lines is None or len(grid_lines) == 0:
            continue

        horizontal_grids, vertical_grids, other_grids = prepare_single_page_grid_lines(grid_lines)

        aligned_grid_lines.append((horizontal_grids, vertical_grids, other_grids))

        h_grids_named, h_grid_name_to_idx = normalize_names(horizontal_grids, 1)
        v_grids_named, v_grid_name_to_idx = normalize_names(vertical_grids, 0)

        grid_sets.append((idx, h_grids_named, v_grids_named, h_grid_name_to_idx, v_grid_name_to_idx))

    h_grid_name_indices, v_grid_name_indices, h_grid_step, v_grid_step = prepare_overall_grids_data(grid_sets)

    relative_orders, aggregated_relative_orders, processed_pages = get_grid_headers_relative_order(grid_sets,
                                                                                              page_names=page_names)

    lost_pages = []
    for i, page_name in enumerate(page_names):
        if i not in processed_pages:
            lost_pages.append(i)


    page_dimensions = [(region[2] - region[0], region[3] - region[1]) for region in page_regions]

    pages_relative_placement = calculate_pages_relative_placement(page_dimensions, aligned_grid_lines,
                                                                  aggregated_relative_orders, relative_orders,
                                                                  grid_sets,
                                                                  h_grid_name_indices, v_grid_name_indices, h_grid_step,
                                                                  v_grid_step,
                                                                  confidence_threshold=0.7, page_names=page_names)

    return pages_relative_placement, lost_pages

def get_stripes_by_placement(direct_placement):
    reverse_placement = 3 - direct_placement  # for left-right settings

    # for top-bottom placement
    if direct_placement == RelativePlacementTypes.BOTTOM.value:
        reverse_placement = RelativePlacementTypes.TOP.value
    elif direct_placement == RelativePlacementTypes.TOP.value:
        reverse_placement = RelativePlacementTypes.BOTTOM.value
    elif direct_placement == RelativePlacementTypes.OVERLAP.value:  # full page intersect
        reverse_placement = direct_placement

    return direct_placement, reverse_placement

def prepare_relative_stripes_data(relative_placement):
    needed_stripes = {}
    placement_dict = {}

    for placement_info in relative_placement:
        p1 = placement_info[0]
        p2 = placement_info[1]
        placements = placement_info[2]  # tuple of primary/secondary placement
        for placement in placements:
            if placement is not None:
                if p1 not in needed_stripes:
                    needed_stripes[p1] = []
                if p2 not in needed_stripes:
                    needed_stripes[p2] = []

                image_one_stripe, image_two_stripe = get_stripes_by_placement(placement)
                if image_one_stripe not in needed_stripes[p1]:
                    needed_stripes[p1].append(image_one_stripe)
                if image_two_stripe not in needed_stripes[p2]:
                    needed_stripes[p2].append(image_two_stripe)

                if (p1, p2) not in placement_dict:
                    placement_dict[(p1, p2)] = []
                if (p2, p1) not in placement_dict:
                    placement_dict[(p2, p1)] = []

                placement_dict[(p1, p2)].append((image_one_stripe, image_two_stripe))
                placement_dict[(p2, p1)].append((image_two_stripe, image_one_stripe))

    return needed_stripes, placement_dict