from copy import deepcopy

import PIL.Image as pil_image
import numpy as np
import cv2
import enum
from src_logging.log_config import setup_logger
logger = setup_logger(__name__)


def detect_stripe_widths(*, match_lines: list[tuple[int, int, int, int]],
                         default_percent: float = 0.3,
                         crop_region: tuple[int, int, int, int] | None,
                         img_arr: np.ndarray) -> list[int] | None:
    stripe_widths = [None] * 5

    if match_lines is not None and len(match_lines) > 0:
        # logger.info('detect_stripe_widths: detecting by match lines')

        for ent in match_lines:
            # logger.info('Match line: %s', ml)
            ml = ent['actual_dashed_line']
            stripe_side = None
            distance_from_edge = None

            region_xl = crop_region[0] if crop_region is not None else 0
            region_yt = crop_region[1] if crop_region is not None else 0

            if ent['type'] == 'horizontal':
                y = int((ml[1] + ml[3]) / 2) - region_yt
                distance_from_edge = min(y, img_arr.shape[0] - y)
                if distance_from_edge > default_percent * img_arr.shape[0] * 0.75:  # stripe width will be x2 of this
                    # logger.info('Match %s line is too far from edge - could be false positive, skipping.', ml)
                    continue
                elif y <= img_arr.shape[0] / 2:
                    stripe_side = ImagePart.TOP_STRIPE.value
                else:
                    stripe_side = ImagePart.BOTTOM_STRIPE.value
            elif ent['type'] == 'vertical':
                x = int((ml[0] + ml[2]) / 2) - region_xl
                distance_from_edge = min(x, img_arr.shape[1] - x)
                if distance_from_edge > default_percent * img_arr.shape[1]:
                    # logger.info('Match line %s is too far from edge - could be false positive, skipping.', ml)
                    continue
                elif x <= img_arr.shape[1] / 2:
                    stripe_side = ImagePart.LEFT_STRIPE.value
                else:
                    stripe_side = ImagePart.RIGHT_STRIPE.value
            else:
                pass
                # logger.info('Match line is not horizontal or vertical - skipping.')

            if stripe_side is not None:
                stripe_widths[stripe_side] = int(distance_from_edge) * 2
                # logger.info('Match line %s allows to determine stripe %s of width %s', ml, stripe_side,
                #             stripe_widths[stripe_side])

    else:
        pass
        # logger.info('detect_stripe_widths: no match lines')

    for stripe_side in [ImagePart.LEFT_STRIPE.value, ImagePart.RIGHT_STRIPE.value]:
        if stripe_widths[stripe_side] is None:
            stripe_widths[stripe_side] = int(default_percent * img_arr.shape[1])

    for stripe_side in [ImagePart.TOP_STRIPE.value, ImagePart.BOTTOM_STRIPE.value]:
        if stripe_widths[stripe_side] is None:
            stripe_widths[stripe_side] = int(default_percent * img_arr.shape[0])

    return stripe_widths


class ImagePart(enum.Enum):
    FULL_PAGE = 0
    LEFT_STRIPE = 1
    RIGHT_STRIPE = 2
    TOP_STRIPE = 3
    BOTTOM_STRIPE = 4


class ImageDataForMatch:
    def __init__(self, original_page_img: np.ndarray, img_arr: np.ndarray,
                 *, match_lines: list[tuple[int, int, int, int]] | None = None, grid_lines: list[dict] | None = None,
                 stripe_width_percent: float | None, crop_region: tuple[int, int, int, int] | None = None):

        self.original_page_img = original_page_img.copy()
        self.img_arr = img_arr.copy()
        self.grid_lines = grid_lines
        self.crop_region = crop_region
        self.match_lines = match_lines
        self.stripe_width_percent = stripe_width_percent

        self.H = np.eye(2, 3)

        if stripe_width_percent is not None:
            stripe_widths = detect_stripe_widths(match_lines=match_lines,
                                                 crop_region=crop_region, img_arr=img_arr,
                                                 default_percent=stripe_width_percent)
        else:
            stripe_widths = None

        if stripe_widths is None and stripe_width_percent is not None:
            # logger.warning(
            #     f"Stripe widths not detected for image {pdf_name} - using default stripe width {stripe_width_percent * 100}%")
            stripe_widths = [0] * 5
            stripe_widths[ImagePart.LEFT_STRIPE.value] = stripe_width_percent * self.img_arr.shape[1]
            stripe_widths[ImagePart.RIGHT_STRIPE.value] = stripe_width_percent * self.img_arr.shape[1]
            stripe_widths[ImagePart.TOP_STRIPE.value] = stripe_width_percent * self.img_arr.shape[0]
            stripe_widths[ImagePart.BOTTOM_STRIPE.value] = stripe_width_percent * self.img_arr.shape[0]

        self.subimages_count = 5 if stripe_widths is not None else 1
        self.keypoints = [None] * self.subimages_count
        self.descriptors = [None] * self.subimages_count
        self.sub_images = [None] * self.subimages_count

        self.sub_images[ImagePart.FULL_PAGE.value] = self.img_arr

        if stripe_widths is not None:
            self.sub_images[ImagePart.LEFT_STRIPE.value] = self.img_arr.copy()
            self.sub_images[ImagePart.LEFT_STRIPE.value][:, int(stripe_widths[ImagePart.LEFT_STRIPE.value]):] = 255

            self.sub_images[ImagePart.RIGHT_STRIPE.value] = self.img_arr.copy()
            self.sub_images[ImagePart.RIGHT_STRIPE.value][:,
            : int(self.img_arr.shape[1] - stripe_widths[ImagePart.RIGHT_STRIPE.value])] = 255

            self.sub_images[ImagePart.TOP_STRIPE.value] = self.img_arr.copy()
            self.sub_images[ImagePart.TOP_STRIPE.value][int(stripe_widths[ImagePart.TOP_STRIPE.value]):, :] = 255

            self.sub_images[ImagePart.BOTTOM_STRIPE.value] = self.img_arr.copy()
            self.sub_images[ImagePart.BOTTOM_STRIPE.value][
            :int(self.img_arr.shape[0] - stripe_widths[ImagePart.BOTTOM_STRIPE.value]),
            :] = 255
        # else:
        #     logger.warning(
        #         f"Stripe widths not detected for image {pdf_name} and default is not set - using full page only.")


def prepare_stitching_image(img_arr,
                            parsed_text,
                            crop_region = None,
                            remove_text=False,
                            threshold=100):
    if not crop_region:
        crop_region = []
    crop_region = list(crop_region)
    # process
    if crop_region:
        original_cropped = np.array(pil_image.fromarray(img_arr).crop(crop_region))
    else:
        original_cropped = deepcopy(img_arr)

    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    if threshold is not None:
        img_arr[img_arr <= threshold] = 255

    # delete text
    if remove_text and parsed_text is not None:
        for text_ent in parsed_text:
            bbox = list(map(int,[text_ent['x0'], text_ent['y0'], text_ent['x1'], text_ent['y1']]))
            img_arr[int(bbox[1]):int(bbox[3]) + 1, int(bbox[0]):int(bbox[2]) + 1] = 255

    img = pil_image.fromarray(img_arr)
    # crop
    if crop_region:
        img = img.crop(crop_region)

    img_arr = np.array(img)

    return original_cropped, img_arr


def find_image_descriptors(img_for_match: ImageDataForMatch, needed_stripes: list[int] | None,
                           sift_octave_layers: int,
                           try_full_page=False):
    for i in range(len(img_for_match.sub_images)):

        # ignore stripe if not needed by grid ordering
        if needed_stripes is not None and i not in needed_stripes:
            img_for_match.keypoints[i], img_for_match.descriptors[i] = [], []
            continue

        # try full page only if it's in needed_stripes or forced by try_full_page or there are no more stripes
        if i == ImagePart.FULL_PAGE.value and not try_full_page and len(img_for_match.sub_images) > 1 and needed_stripes is None:
            img_for_match.keypoints[i], img_for_match.descriptors[i] = [], []
            continue

        feature_detector = cv2.SIFT_create(nOctaveLayers=sift_octave_layers, enable_precise_upscale=True)

        img_for_match.keypoints[i], img_for_match.descriptors[i] = feature_detector.detectAndCompute(
            img_for_match.sub_images[i], None)


def filter_insane_gridlines_for_page(grid_lines, page_width, page_height, *, min_line_length=100):
    resulting_grids = []

    used_grid_names = dict()

    for grid_line_data in grid_lines:
        text = grid_line_data['text'].strip().lower()
        if text not in used_grid_names:
            used_grid_names[text] = 0
        used_grid_names[text] += 1

    for grid_line_data in grid_lines:
        # bbox = grid_line_data['bbox']
        text = grid_line_data['text'].strip().lower()
        line = grid_line_data['grid']

        if used_grid_names[text] > 1:
            logger.warning('Grid line %s is duplicated %s times!', text, used_grid_names[text])
            continue

        line_len = max(abs(line[2] - line[0]), abs(line[3] - line[1]))
        if line_len < min_line_length:  # ? use page_width and page_height and relative length?
            logger.warning('Grid line %s is too short: %s', text, line_len)
            continue

        resulting_grids.append(grid_line_data)

    return resulting_grids


def fix_grid_lines_data(data: list[dict], *, fix_disorder: bool = True) -> list[dict]:
    for line in data:
        dx = line['grid'][2] - line['grid'][0]
        dy = line['grid'][3] - line['grid'][1]

        is_v = abs(dx) < abs(dy)

        if fix_disorder:
            if is_v:
                if line['grid'][1] > line['grid'][3]:
                    line['grid'][1], line['grid'][3] = line['grid'][3], line['grid'][1]
                    line['grid'][0], line['grid'][2] = line['grid'][2], line['grid'][0]
            else:
                if line['grid'][0] > line['grid'][2]:
                    line['grid'][0], line['grid'][2] = line['grid'][2], line['grid'][0]
                    line['grid'][1], line['grid'][3] = line['grid'][3], line['grid'][1]

    if fix_disorder:
        data.sort(key=lambda x: x['grid'][0] if abs(x['grid'][0] - x['grid'][2]) > abs(x['grid'][1] - x['grid'][3]) else
        x['grid'][1])

    return data

def filter_insane_gridlines(grid_lines, pages_dimensions, *, min_line_length=100):
    resulting_grid_lines = []
    for i, page_grids in enumerate(grid_lines):
        if page_grids is None:
            resulting_grid_lines.append(None)
            continue

        page_width, page_height = pages_dimensions[i]

        resulting_grid_lines.append(filter_insane_gridlines_for_page(page_grids, page_width, page_height,
                                                                     min_line_length=min_line_length))

    return resulting_grid_lines


def update_bounding_box(bbox: list[int, int, int, int], x1: int, y1: int, x2: int, y2: int) -> tuple[
    int, int, int, int]:
    if bbox[0] < 0 or bbox[0] > min(x1, x2):
        bbox[0] = min(x1, x2)

    bbox[2] = max(x1, x2, bbox[2])

    if bbox[1] < 0 or bbox[1] > min(y1, y2):
        bbox[1] = min(y1, y2)

    bbox[3] = max(y1, y2, bbox[3])


def get_page_bounding_box(grid_lines_data: list[dict]) -> tuple[int, int, int, int]:
    """Get page bounding box from grid lines data"""
    x_min1 = min([line['bbox'][0] for line in grid_lines_data])
    x_min1 = min(min([line['grid'][0] for line in grid_lines_data]), x_min1)
    x_min1 = min(min([line['bbox'][2] for line in grid_lines_data]), x_min1)
    x_min1 = min(min([line['grid'][2] for line in grid_lines_data]), x_min1)

    y_min1 = min([line['bbox'][1] for line in grid_lines_data])
    y_min1 = min(min([line['grid'][1] for line in grid_lines_data]), y_min1)
    y_min1 = min(min([line['bbox'][3] for line in grid_lines_data]), y_min1)
    y_min1 = min(min([line['grid'][3] for line in grid_lines_data]), y_min1)

    y_max = max([line['bbox'][1] for line in grid_lines_data])
    y_max = max(max([line['grid'][1] for line in grid_lines_data]), y_max)
    y_max = max(max([line['bbox'][3] for line in grid_lines_data]), y_max)
    y_max = max(max([line['grid'][3] for line in grid_lines_data]), y_max)

    x_max = max([line['bbox'][0] for line in grid_lines_data])
    x_max = max(max([line['bbox'][2] for line in grid_lines_data]), x_max)

    return x_min1, y_min1, x_max, y_max


def detect_crop_region_by_grids(image_width, image_height, grid_lines: list, *, minimal_area_proportion: float = 0.5) -> \
        tuple[int, int, int, int] | None:
    crop_region = None
    try:
        if grid_lines is not None and len(grid_lines) > 0:
            crop_region = get_page_bounding_box(grid_lines)
            s1 = abs(crop_region[0] - crop_region[2]) * abs(crop_region[1] - crop_region[3])
            s2 = image_width * image_height
            if s1 < minimal_area_proportion * s2:
                crop_region = None
    except Exception as e:
        pass

    return crop_region


def get_grid_based_page_bounds(original_grid_lines: list[list[dict] | None], page_dimensions: list[tuple[int, int]], *,
                               use_label_box_only: bool = True, minimal_area_proportion: float = 0.5) -> \
        list[tuple[int, int, int, int] | None]:
    result = []

    for i, page_grids in enumerate(original_grid_lines):
        if page_grids is None:
            result.append(None)
            continue

        w = page_dimensions[i][0]
        h = page_dimensions[i][1]
        page_bbox = detect_crop_region_by_grids(w, h, page_grids)

        result.append((page_bbox[0], page_bbox[1], page_bbox[2], page_bbox[3]) if page_bbox is not None else None)

    return result


def get_suitable_segmentation_for_pages(original_segmentations_data: list[dict | None], pages_dimensions, *,
                                        area_proportion_threshold=0.25) -> list[tuple[int, int, int, int]]:
    result = []
    for i, pages_dimension in enumerate(pages_dimensions):
        page_segmentations = original_segmentations_data[i] if original_segmentations_data is not None else None
        image_width, image_height = pages_dimension

        try:
            page_largest_segment = None
            if page_segmentations is not None and len(page_segmentations) > 0:
                if len(page_segmentations) > 1:
                    page_largest_segment = max(page_segmentations,
                                               key=lambda x: abs(x['segmented_area'][0] - x['segmented_area'][2]) * abs(
                                                   x['segmented_area'][1] - x['segmented_area'][3]))
                else:
                    page_largest_segment = page_segmentations[0]['segmented_area'] if len(
                        page_segmentations) > 0 else None

                s = abs((page_largest_segment[2] - page_largest_segment[0]) * (
                        page_largest_segment[3] - page_largest_segment[1]))

                if s < image_width * image_height * area_proportion_threshold:
                    page_largest_segment = None
        except:
            page_largest_segment = None

        result.append(page_largest_segment)

    return result


def select_crop_regions(segmentation_for_pages: list[tuple[int, int, int, int] | None],
                        page_bounds_by_grids: list[tuple[int, int, int, int] | None],
                        original_page_dimensions: list[tuple[int, int]]) -> list[tuple[int, int, int, int] | None]:
    result = []
    for i, original_page_dimension in enumerate(original_page_dimensions):
        if segmentation_for_pages and segmentation_for_pages[i] is not None:
            result.append(segmentation_for_pages[i])
        elif page_bounds_by_grids and page_bounds_by_grids[i] is not None:
            result.append(page_bounds_by_grids[i])
        else:
            result.append((0, 0, original_page_dimension[0], original_page_dimension[1]))

    return result
