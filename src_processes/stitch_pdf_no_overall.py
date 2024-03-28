import fitz
import numpy as np

from src_logging.log_config import setup_logger
from src_utils.feature_generation import prepare_stitching_image, ImageDataForMatch, find_image_descriptors, \
    filter_insane_gridlines, get_grid_based_page_bounds, get_suitable_segmentation_for_pages, select_crop_regions, \
    fix_grid_lines_data
from src_utils.geometry_utils import point_cw_rotate, fix_coords, scale_crop
from src_utils.matching_utils import find_stitching_transformations, get_pairwise_transformations_by_grids, \
    get_stitching_sequence, calculate_final_transformations, get_final_placements_for_pages
from src_utils.page_ordering_utils import get_pages_relative_placement, prepare_relative_stripes_data
from src_utils.pdf_utils import create_empty_doc_w_page, crop_page, insert_blank_image, fix_cropbox
from src_utils.postprocessing_utils import calculate_overlap_regions_data
logger = setup_logger(__name__)


def prepare_stitching_data_dictionary(source_regions, resulting_regions, overall_w, overall_h) -> dict:
    pages_data = []
    for page_idx, source_region in enumerate(source_regions):
        pages_data.append({
            "pdf_name": 'page_' + str(page_idx+1),
            "image_rect": resulting_regions[page_idx],
            "crop_region": source_region
        })

    result = {
        "overall_region": [0, 0, overall_w, overall_h],
        "pages": pages_data,
        "pdf_set_name": 'pdf_set_for_stitching'
    }

    return result

def stitch_pdf_no_overall(docs_list,
                          images_list,
                          parsed_text_list,
                          match_lines_list,
                          crop_regions_list,
                          grid_lines_list,
                          remove_text,
                          config):

    ## Prepare all non-pixel data
    original_page_dimentions = [(img.shape[1], img.shape[0]) for img in images_list]

    grid_lines_fixed = [fix_grid_lines_data(page_grid_lines) for page_grid_lines in grid_lines_list] if grid_lines_list is not None else None
    usable_grid_lines = filter_insane_gridlines(grid_lines_fixed, original_page_dimentions) if grid_lines_list is not None else None

    page_bounds_by_grids = get_grid_based_page_bounds(grid_lines_fixed, original_page_dimentions) if grid_lines_list is not None else None

    segmentation_for_pages = get_suitable_segmentation_for_pages(crop_regions_list, original_page_dimentions)

    pages_crop_regions = select_crop_regions(segmentation_for_pages, page_bounds_by_grids, original_page_dimentions)

    original_cropped_images = []
    prepared_images = []

    for c, img in enumerate(images_list):
        original_cropped, prepared_image = prepare_stitching_image(img_arr=img,
                                                                   parsed_text=parsed_text_list[c]['parsed_text'] if parsed_text_list else None,
                                                                   crop_region=pages_crop_regions[c],
                                                                   remove_text=remove_text,
                                                                   **config['prepare_stitching_image'])
        original_cropped_images.append(original_cropped)
        prepared_images.append(prepared_image)

    # try stitch pages using grid lines only. Page images are not used on this stage
    # config parameters: intersection_area_threshold, min_intersection_len_threshold, scale_threshold, rotation_threshold
    if usable_grid_lines:
        pairwise_transformations_by_grids = get_pairwise_transformations_by_grids(pages_crop_regions, usable_grid_lines,
                                                                         original_page_dimentions,
                                                                         **config['get_pairwise_transformations_by_grids'])

        logger.info('Found %s grid based pairwise transformations: %s',
                    int(len(pairwise_transformations_by_grids.keys()) / 2), pairwise_transformations_by_grids.keys())

        stitching_sequence = get_stitching_sequence(pairwise_transformations_by_grids)
        logger.info('Stitching sequence calculated by grids: %s', stitching_sequence)
    else:
        stitching_sequence = []
        pairwise_transformations_by_grids = None

    if stitching_sequence and (len(stitching_sequence) == len(docs_list) - 1):
        logger.info("Grid-based stitching sequence is complete - no need to use image matching approach.")

        processed_pages_count = len(stitching_sequence)

        final_transformations, stitched_pages_indices = calculate_final_transformations(stitching_sequence,
                                                                                        pairwise_transformations_by_grids,
                                                                                        pages_crop_regions)
        logger.info('Final grid based transformations calculated: %s',
                    [(t[0][2], t[1][2]) for t in final_transformations.values()])

        resulting_regions, w, h = get_final_placements_for_pages(final_transformations, pages_crop_regions)
        logger.info('Final grid based placements calculated. Overall drawing size: %s; page regions: %s', (w, h),
                    resulting_regions)

        resulting_stitch_data = prepare_stitching_data_dictionary(pages_crop_regions, resulting_regions, w, h)
        overall_width, overall_height = w, h
        transformations = resulting_stitch_data['pages']
    else:
        logger.info(
            "Grid-based stitching sequence is incomplete - grid-based ordering and key points will be used to complete the stitching.")

        # try to calculate relative placement
        page_names = [f'page_to_stitch_{i}' for i in range(0, len(docs_list))]

        try:
            relative_placement, lost_pages = get_pages_relative_placement(page_names, usable_grid_lines,
                                                                          pages_crop_regions, update_grids_by_crop=True)
            logger.info('Relative placements calculated: %s. Lost pages (no information): %s', len(relative_placement),
                        lost_pages)
        except Exception as e:
            logger.exception("Exception getting relative placement by grids! %s", e)
            lost_pages = page_names
            relative_placement = None

        needed_stripes = placement_dict = None
        if lost_pages and len(lost_pages) > 0:
            logger.error("There are lost pages - grid based relative placement will not be used! Lost pages: %s",
                         lost_pages)
        elif relative_placement is not None:
            needed_stripes, placement_dict = prepare_relative_stripes_data(relative_placement)
        else:
            logger.error("No relative placement found by grid lines!")

        ## process images and find descriptors
        processed_images = []
        for c, img in enumerate(images_list):
            image_for_match = ImageDataForMatch(original_page_img=original_cropped_images[c],
                                img_arr=prepared_images[c],
                                stripe_width_percent=config['find_stitching_transformations']['stripe_width_percent'],
                                match_lines=match_lines_list[c] if match_lines_list[c] else None,
                                                grid_lines=grid_lines_fixed[c] if grid_lines_fixed[c] else None,
                                                crop_region=pages_crop_regions[c])

            find_image_descriptors(image_for_match, needed_stripes[c] if needed_stripes else None, **config['find_image_descriptors'])
            processed_images.append(image_for_match)

        logger.info('Processed images, found descriptors')

        ## here we get information about each area transformation in order

        overall_region, transformations = find_stitching_transformations(processed_images, placement_dict,
                                                                         pairwise_transformations_by_grids,
                                                                         **config['find_stitching_transformations'])
        logger.info(f'Transformations : {transformations}')
        logger.info(f'Overall region : {overall_region}')

        if not transformations:
            raise Exception('No matches found')

        overall_width, overall_height = overall_region[2:]
        processed_pages_count = len(processed_images)

        logger.info('Estimated transformations by keypoints.')

    ## stitching routine
    ## calculate overlap regions data
    if processed_pages_count<4:
        pdf_page_pixels = [img_arr for img_arr in prepared_images]
        overlap_regions = calculate_overlap_regions_data(transformations,
                                       pdf_page_pixels=pdf_page_pixels,
                                       **config['calculate_overlap_regions_data'])
    else:
        overlap_regions = None

    ### find rotation
    rotation_set = []
    for doc in docs_list:
        # load file
        page = doc[0]
        # rotation routine
        rotation = page.rotation
        rotation_set.append(rotation)

    rotation_set = list(set([i for i in rotation_set if i != 0]))
    if rotation_set:
        rotation = rotation_set[0]
    else:
        rotation = 0

    if rotation in [90, 270]:
        overall_height, overall_width = overall_width, overall_height

    logger.info(f'Identified rotation : {rotation}')

    ### rotate data
    for c, doc in enumerate(docs_list):
        # load file
        page = doc[0]
        # rotation routine
        page.set_rotation(0)
        # get pages info
        info_to_use = transformations[c]
        logger.info(info_to_use)

        cropbox = info_to_use['crop_region']
        page_rect = info_to_use['image_rect']
        # rotate pages info
        info_to_use['image_rect'] = fix_coords([*point_cw_rotate(page_rect[:2], rotation, [0, 0, overall_width, overall_height]), \
                                                *point_cw_rotate(page_rect[2:], rotation, [0, 0, overall_width, overall_height])])
        if cropbox:
            info_to_use['crop_region'] = fix_coords([*point_cw_rotate(cropbox[:2], rotation, page.rect), \
                                                 *point_cw_rotate(cropbox[2:], rotation, page.rect)])



    logger.info('Rotated data')
    ### stitching itself
    empty_overall_doc = create_empty_doc_w_page(width=overall_width,
                                                height=overall_height)
    empty_overall_doc_page = empty_overall_doc[0]
    original_sizes = []
    changed_sizes = []
    transformation_matrices = []
    for c, doc in enumerate(docs_list):
        # get pdf
        page = doc[0]
        page.set_rotation(0)
        page_w, page_h = page.rect.br
        original_sizes.append((page_w, page_h))
        # get info
        info_to_use = transformations[c]
        cropbox = info_to_use['crop_region']
        page_rect = info_to_use['image_rect']
        # crop page
        if cropbox:
            # fix cropbox
            if list(page.rect)!=list(page.mediabox):
                cropbox = fix_cropbox(cropbox, page.transformation_matrix)
                transformation_matrices.append(page.transformation_matrix)
            else:
                transformation_matrices.append(None)
            crop_page(doc, page, cropbox, page_w, page_h)
        changed_sizes.append((page.rect.br[0],page.rect.br[1]))
        # get area for insertion
        page_rect = list(map(lambda x: int(np.round(x)), page_rect))
        empty_overall_doc_page.show_pdf_page(fitz.Rect(page_rect), doc, 0)

    logger.info('Stitched PDFs')
    if overlap_regions:
        for overlap_region in overlap_regions:
            # get info for overlap region
            overlay_crop_region = overlap_region['crop_region']
            overlay_page_rect = overlap_region['image_rect']

            page_idx = overlap_region['page_index']
            # get page
            doc = docs_list[page_idx]
            page = doc[0]
            # rotate crop regions and overlay page rects
            overlay_page_rect = fix_coords(
                [*point_cw_rotate(overlay_page_rect[:2], rotation,
                                  [0, 0, overall_width, overall_height]), \
                 *point_cw_rotate(overlay_page_rect[2:], rotation,
                                  [0, 0, overall_width, overall_height])])

            overlay_page_rect = list(map(lambda x: int(np.round(x)), overlay_page_rect))

            overlay_crop_region = fix_coords([*point_cw_rotate(overlay_crop_region[:2], rotation, [0,0, *changed_sizes[page_idx]]), \
                                                 *point_cw_rotate(overlay_crop_region[2:], rotation, [0,0, *changed_sizes[page_idx]])])

            # scale crop for new crop region

            overlay_crop_region = scale_crop(overlay_crop_region,
                       transformations[page_idx]['crop_region'],
                       original_sizes[page_idx]
                       )

            # insert blank image to overlay region
            insert_blank_image(empty_overall_doc_page,
                               overlay_page_rect,
                               width=int(abs(overlay_crop_region[2]-overlay_crop_region[0])),
                               height=int(abs(overlay_crop_region[3]-overlay_crop_region[1])))

            # crop page_copy
            if overlay_crop_region:
                overlay_crop_region = fix_cropbox(overlay_crop_region,
                                                  transformation_matrices[page_idx])

                crop_page(doc, page, overlay_crop_region,
                          page_w, page_h, False)
            empty_overall_doc_page.show_pdf_page(fitz.Rect(overlay_page_rect), doc, 0)

    empty_overall_doc_page.set_rotation(rotation)

    logger.info('Post-processed results, deleted overlaps')
    return empty_overall_doc.write()
