import numpy as np
from src_logging.log_config import setup_logger
logger = setup_logger(__name__)
def calculate_overlap_regions_data(stitching_data: dict, *, pdf_page_pixels: list[list] | None = None,
                                   non_white_pixels_threshold: int = 200) -> list[dict]:
    overlap_regions = []

    for idx, page in enumerate(stitching_data):
        image_rect = page['image_rect']

        if image_rect[0] > image_rect[2]:
            image_rect[0], image_rect[2] = image_rect[2], image_rect[0]

        if image_rect[1] > image_rect[3]:
            image_rect[1], image_rect[3] = image_rect[3], image_rect[1]


        for idx2 in range(idx + 1, len(stitching_data)):
            page2 = stitching_data[idx2]
            image_rect2 = page2['image_rect']

            if image_rect2[0] > image_rect2[2]:
                image_rect2[0], image_rect2[2] = image_rect2[2], image_rect2[0]

            if image_rect2[1] > image_rect2[3]:
                image_rect2[1], image_rect2[3] = image_rect2[3], image_rect2[1]


            # find the intersection of the two images
            intersection = [max(image_rect[0], image_rect2[0]), max(image_rect[1], image_rect2[1]),
                            min(image_rect[2], image_rect2[2]), min(image_rect[3], image_rect2[3])]

            # if there is no intersection, continue
            if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
                continue

            is_vertical = False
            if intersection[2] - intersection[0] < intersection[3] - intersection[1]:
                is_vertical = True

            # split the intersection into two parts along the longer side of the intersection
            # and find regions of image_1 and image_2 to "restore" after overlapping the second image over the first
            split_line = [intersection[0], intersection[1], intersection[2], intersection[3]]
            if is_vertical:
                split_line[2] = split_line[0] = (intersection[0] + intersection[2]) // 2
                if image_rect[0] < image_rect2[0]:
                    image_1_part = [intersection[0], intersection[1], split_line[2], intersection[3]]
                    image_2_part = [split_line[0], intersection[1], intersection[2], intersection[3]]
                else:
                    image_1_part = [split_line[0], intersection[1], intersection[2], intersection[3]]
                    image_2_part = [intersection[0], intersection[1], split_line[2], intersection[3]]
            else:
                split_line[3] = split_line[1] = (intersection[1] + intersection[3]) // 2
                if image_rect[1] < image_rect2[1]:
                    image_1_part = [intersection[0], intersection[1], intersection[2], split_line[3]]
                    image_2_part = [intersection[0], split_line[1], intersection[2], intersection[3]]
                else:
                    image_1_part = [intersection[0], split_line[1], intersection[2], intersection[3]]
                    image_2_part = [intersection[0], intersection[1], intersection[2], split_line[3]]


            img1_part_source = [image_1_part[0] - image_rect[0], image_1_part[1] - image_rect[1],
                                image_1_part[2] - image_rect[0], image_1_part[3] - image_rect[1]]
            img2_part_source = [image_2_part[0] - image_rect2[0], image_2_part[1] - image_rect2[1],
                                image_2_part[2] - image_rect2[0], image_2_part[3] - image_rect2[1]]


            # TODO: check - because of scaling (even minimal) there are sometimes several pixels difference
            #  in the sizes of crop and target rectangles. Decide - should it be handled here or later during actual
            #  cropping and pasting of the images

            # calculated data for the pair of pages
            calculated_regions_data = [
                {'page_index': idx, 'crop_region': img1_part_source,
                 'image_rect': image_1_part},
                {'page_index': idx2, 'crop_region': img2_part_source,
                 'image_rect': image_2_part}
            ]
            if pdf_page_pixels is not None and len(pdf_page_pixels) > 0:
                indices = [idx, idx2]
                for i, page_idx in enumerate(indices):
                    page_img = pdf_page_pixels[page_idx]

                    source_region = calculated_regions_data[i]['crop_region']

                    pixels_in_region = page_img[source_region[1]:source_region[3], source_region[0]:source_region[2]]

                    # calculate the percentage of non-white pixels in the region
                    non_white_pixels = np.sum(pixels_in_region < non_white_pixels_threshold)
                    # total_pixels = pixels_in_region.shape[0] * pixels_in_region.shape[1]
                    # non_white_percentage = non_white_pixels / total_pixels * 100
                    calculated_regions_data[i]['region_weight'] = non_white_pixels

            # For each page there can be more than one region, if it overlaps with more than one page
            overlap_regions.extend(calculated_regions_data)

        overlap_regions.sort(key=lambda x: x['region_weight'] if 'region_weight' in x else 1.0, reverse=True)
    return overlap_regions