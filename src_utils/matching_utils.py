import math
from copy import deepcopy
from typing import Any
import networkx as nx
import numpy as np
import cv2
from src_utils.feature_generation import ImageDataForMatch, ImagePart
from src_utils.geometry_utils import get_points_of_intersection_scale, get_page_rects_intersection_info
from fastapi import HTTPException

def update_top_best(best_items: list[tuple], new_item: tuple, max_items: int, comparer: callable) -> None:
    if len(best_items) < max_items:
        best_items.append(new_item)
        best_items.sort(key=lambda x: comparer(x), reverse=True)
    else:
        worst_best_item = best_items[-1]
        if comparer(worst_best_item) < comparer(new_item):
            best_items[-1] = new_item
            best_items.sort(key=lambda x: comparer(x), reverse=True)


def knn_matcher(image_data_for_match1: ImageDataForMatch,
                image_data_for_match2: ImageDataForMatch,
                k: int,
                l: int,
                lowes_thr=0.7) -> tuple[list[Any], int, int, int] | tuple[
    list[list[Any]], int | Any, float | int | Any, int | Any]:
    des1 = image_data_for_match1.descriptors[k]
    des2 = image_data_for_match2.descriptors[l]

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return [], 0, 0, 0

    bf = cv2.BFMatcher()

    match = bf.knnMatch(des1, des2, k=2)

    # filter matches by D. Lowe's ratio test
    good = []
    for m, n in match:
        if m.distance < lowes_thr * n.distance:
            good.append([m])

    # TODO: remove worst 10% of matches by distance?
    remove_worst_percent = 0.0
    dists: np.ndarray = np.array([m[0].distance for m in good])
    dists.sort()
    dists = dists[:int(len(dists) * (1 - remove_worst_percent))]
    if len(dists) > 1:
        median_distance = np.median(dists)
        total_distance = np.sum(dists)
        avg_distance = total_distance / len(dists)

    else:
        median_distance = 0
        total_distance = 0
        avg_distance = 0

    return good, median_distance, avg_distance, total_distance


def find_best_translation_for_inliers(src_pts, dst_pts, inliers):
    src_pts_matched = src_pts[inliers.ravel() == 1]
    dst_pts_matched = dst_pts[inliers.ravel() == 1]

    x_y_src_pairs = list(zip(src_pts_matched[:, 0, 0], src_pts_matched[:, 0, 1]))
    x_y_dst_pairs = list(zip(dst_pts_matched[:, 0, 0], dst_pts_matched[:, 0, 1]))

    A = np.zeros((len(x_y_src_pairs) * 2, 2))
    B = np.zeros((len(x_y_src_pairs) * 2, 1))

    for pidx, (x_src, y_src) in enumerate(x_y_src_pairs):
        A[pidx * 2, 0] = 1
        A[pidx * 2, 1] = 0
        A[pidx * 2 + 1, 0] = 0
        A[pidx * 2 + 1, 1] = 1

        B[pidx * 2, 0] = -x_y_dst_pairs[pidx][0] + x_src
        B[pidx * 2 + 1, 0] = -x_y_dst_pairs[pidx][1] + y_src

    # solve the system
    X = np.linalg.lstsq(A, B, rcond=None)[0]

    # build a translation matrix
    T = np.array([[1, 0, round(X[0][0])], [0, 1, round(X[1][0])]], dtype=np.float32)
    return T

def find_image_matches(prepared_data: list[ImageDataForMatch], *, placement_dict, stripe_width_percent: float,
                       minimum_matches_count: int,  # if there are less matchies, ignore variant
                       minimum_inliers_count: int,  # if there are less inliers, ignore variant
                       inliers_threshold: float = 0.2,
                       # if inliers count is less than this percent of all matches, ignore variant
                       inliers_to_best_threshold: float = 0.7,
                       # if we have match, even rejected, with much more inliers - ignore this one
                       intersection_area_threshold=0.45,
                       # if intersection area is more than this percent of the first page area - ignore variant
                       min_intersection_len_threshold=0.35,
                       # biggest side of intersection rectangle is less than this percent of the first page side - ignore variant
                       maximum_rotation: float,
                       # if rotation is more than this - ignore variant. ? Allow rotation near 90?
                       lowes_thr: float,
                       try_full_page: bool  # Try match full page to full page also, not only stripes
                       ) -> tuple[dict, list]:
    N = len(prepared_data)

    good_matches = {}
    match_counts = []
    for i in range(N):
        for j in range(i + 1, N):
            max_inliers_count = 0

            if placement_dict is not None and (i, j) not in placement_dict:  # no such pair to match
                if prepared_data[i].grid_lines is not None and len(prepared_data[i].grid_lines) > 0 and prepared_data[
                    j].grid_lines is not None and len(prepared_data[j].grid_lines) > 0:
                    # No placement info for pages i and j, but both have grid lines - skipping pair
                    continue

            start_subimage_index = 1 if (stripe_width_percent is not None and not try_full_page) else 0

            if placement_dict is not None and (i, j) in placement_dict and placement_dict[(i, j)] == 0:
                # Matching full to full for pages i and j
                start_subimage_index = 0

            best_matches = []
            skipped_stripes = 0
            for k in range(start_subimage_index, len(prepared_data[i].sub_images)):
                for l in range(start_subimage_index, len(prepared_data[j].sub_images)):
                    if placement_dict is not None and (i, j) in placement_dict:  # there is a placement info
                        if (k, l) not in placement_dict[(i, j)]:
                            # print(f"No placement info for stripes {k} and {l} for pages {i} and {j} - skipping stripes pair.")
                            # logger.info("Skipping stripes %s and %s for pages %s and %s, as for match type %s we need only stripes %s vs %s", k, l, page1_name, page2_name, match_side_code, stripe_to_check_1, stripe_to_check_2)
                            skipped_stripes += 1
                            continue

                    matches, median_distance, avg_distance, total_distance = knn_matcher(prepared_data[i],
                                                                                         prepared_data[j], k, l,
                                                                                         lowes_thr=lowes_thr)
                    if len(matches) < minimum_matches_count:
                        continue

                    # estimate transformation and check if it's feasible and has enough inliers
                    kp1 = prepared_data[i].keypoints[k]
                    kp2 = prepared_data[j].keypoints[l]

                    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    H = None
                    try:
                        H, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)
                    except:
                        # log error?
                        pass

                    if H is None:
                        continue

                    inliers_count = int(len(inliers[inliers == 1]))

                    if inliers_count > max_inliers_count:
                        max_inliers_count = inliers_count

                    # This check is not enough - it's only some optimization. Will be repeated after all matches are checked for the pages pair.
                    if inliers_count < max_inliers_count * inliers_to_best_threshold:
                        continue

                    if inliers_count < minimum_inliers_count:
                        continue

                    inliers_koef = inliers_count / len(inliers)
                    if inliers_koef < inliers_threshold:
                        continue

                    # get match score
                    match_score = math.log2(inliers_count) + inliers_koef * 5

                    # scale = np.sqrt(H[0][0] * H[0][0] + H[0][1] * H[0][1])
                    rot = np.arctan2(H[1][0], H[0][0]) * 180 / np.pi
                    # x_shift = H[0][2]
                    # y_shift = H[1][2]

                    if maximum_rotation is not None and (
                            abs(rot) > maximum_rotation and abs(abs(rot) - math.pi / 2) > maximum_rotation):
                        # Rotation is too big - match will be used as the last resort only - decrease score
                        match_score /= 1.5

                    # replace H with pure translation matrix which suits inliers the best in terms of least sqaures
                    H = find_best_translation_for_inliers(src_pts, dst_pts, inliers)

                    img1 = prepared_data[i].sub_images[ImagePart.FULL_PAGE.value]
                    img2 = prepared_data[j].sub_images[ImagePart.FULL_PAGE.value]

                    img1_rect = np.array(
                        [[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]]],
                        dtype=np.float32)
                    img2_rect = np.array(
                        [[0, 0], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]], [0, img2.shape[0]]],
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
                    img1_area = img1.shape[1] * img1.shape[0]

                    # Compute the proportion of the intersection area to the area of the first image
                    area_proportion = intersection_area / img1_area
                    side_length_proportion = max(intersection_width / img1.shape[1],
                                                 intersection_height / img1.shape[0])

                    if area_proportion > intersection_area_threshold:
                        continue

                    if side_length_proportion < min_intersection_len_threshold:
                        continue


                    update_top_best(best_matches, (matches, inliers_count, k, l, H, inliers, match_score),
                                    3, lambda x: x[1])

            all_rotated = True  # Sometimes we have only rotated matches - let's preserve it if we have no other matches
            good_matches[(i, j)] = []
            # go for all matches between a page pair and do checks which require statistics for all matches between the pair
            for m_num, good_match in enumerate(best_matches):
                matches, inliers_count, k, l, H, inliers, match_score = good_match
                if inliers_count < max_inliers_count * inliers_to_best_threshold:
                    continue

                if not (maximum_rotation is not None and (
                        abs(rot) > maximum_rotation and abs(abs(rot) - math.pi / 2) > maximum_rotation)):
                    all_rotated = False

                good_matches[(i, j)].append(good_match)

            # If there are non-rotated matches, remove rotated ones
            if not all_rotated:
                good_matches[(i, j)] = [good_match for good_match in good_matches[(i, j)] if
                                        abs(rot) <= maximum_rotation or abs(abs(rot) - math.pi / 2) <= maximum_rotation]

            if len(good_matches[(i, j)]) > 0:
                matches_count = len(good_matches[(i, j)][0][0])
                match_counts.append(matches_count)
            else:
                match_counts.append(0)

    return good_matches, match_counts


def calculate_pairwise_transformations(good_matches: dict,
                                       match_counts: list, matches_against_median_threshold: float,
                                       inliers_threshold: float,
                                       stripe_width_percent: float) -> list:
    pairwise_transformations = []
    for i, j in good_matches.keys():

        if len(good_matches[(i, j)]) == 0:
            continue

        best_match = good_matches[(i, j)][0]

        matched_points, inliers_count, img1_stripe_id, img2_stripe_id, H, inliers, match_score = best_match
        inliers_percent = inliers_count / len(inliers) * 100

        # if we match only full page to full page, filter out combinations with significantly less matches than median
        # not needed with other sanity checks now?
        rules = [stripe_width_percent is None and len(
            matched_points) < np.median(match_counts) * matches_against_median_threshold,
                 inliers_percent < inliers_threshold * 100,
                 H is None]

        # also if we will allow rotation and scale, we can decrease score for matches with big rotation
        # now we have translation only matrix from matcher

        if any(rules):
            continue

        pairwise_transformations.append((i, j, H, img1_stripe_id, img2_stripe_id, match_score))

    return pairwise_transformations


def stitch_next_image(prepared_data: list[ImageDataForMatch], overall_image: np.ndarray, source_H: np.ndarray,
                      target_image_idx: int,
                      target_to_source_H: np.ndarray
                      ):
    img2 = prepared_data[target_image_idx].original_page_img

    w1 = overall_image.shape[1]
    h1 = overall_image.shape[0]

    w2 = img2.shape[1]
    h2 = img2.shape[0]

    xr1 = w1
    yb1 = h1

    image_2_rect = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]],
                            dtype=np.float32)

    H = deepcopy(target_to_source_H)

    H_extended = np.zeros((3, 3))
    H_extended[0:2, :] = H
    H_extended[2, 2] = 1

    source_H_extended = np.zeros((3, 3))
    source_H_extended[0:2, :] = source_H
    source_H_extended[2, 2] = 1

    combined_H = np.matmul(source_H_extended, H_extended)

    image_2_rect_transformed = cv2.transform(np.array([image_2_rect]), combined_H)[0]

    # remove the 3rd row
    combined_H = combined_H[0:2, :]

    xl2 = round(image_2_rect_transformed[0][0])
    yt2 = round(image_2_rect_transformed[0][1])
    xr2 = round(image_2_rect_transformed[2][0])
    yb2 = round(image_2_rect_transformed[2][1])

    sx = round(max(-xl2, 0))
    sy = round(max(-yt2, 0))

    new_w = max(xr1 + sx, xr2 + sx)
    new_h = max(yb1 + sy, yb2 + sy)

    # white image of the size of stitched image
    image_2_for_stitch = np.ones((new_h, new_w, 3), dtype=np.uint8)

    final_H = deepcopy(combined_H)
    # # shift all if new coordinates are negative
    final_H[0][2] += sx
    final_H[1][2] += sy

    overall_H = np.eye(2, 3)

    overall_H[0][2] += sx
    overall_H[1][2] += sy

    cv2.warpAffine(img2, final_H, (new_w, new_h), image_2_for_stitch)

    image_1_for_stitch = np.ones((new_h, new_w, 3), dtype=np.uint8)
    image_1_for_stitch[sy:h1 + sy, sx:w1 + sx, :] = overall_image

    alpha = 0.5  # blending factor
    stitched_image = cv2.addWeighted(image_2_for_stitch, alpha, image_1_for_stitch, 1 - alpha, 0)

    return stitched_image, overall_H, final_H


def stitch_images_overall(pairwise_transformations: list[tuple], prepared_data: list[ImageDataForMatch]
                          ):
    stitching_sequence_graph = nx.Graph()
    for i, j, H, img1_stripe_id, img2_stripe_id, match_score in pairwise_transformations:
        src_H = np.eye(2, 3)

        stitching_sequence_graph.add_edge(i, j, first_idx=i, second_idx=j, H1=src_H, H2=H,
                                          img1_stripe_id=img1_stripe_id,
                                          img2_stripe_id=img2_stripe_id, weight=1.0 / match_score)

    stitching_sequence_graph = nx.minimum_spanning_tree(stitching_sequence_graph, weight='weight')

    start_page_idx = 0

    # check if start_page_idx is in the graph
    while start_page_idx not in stitching_sequence_graph and start_page_idx < len(prepared_data):
        start_page_idx += 1

    if start_page_idx >= len(prepared_data):
        return None, None

    stitch_sequence = list(nx.bfs_edges(stitching_sequence_graph, start_page_idx))

    overall_stitched_image = None
    added_pages_transforms = {}
    for source_image_idx, target_image_idx in stitch_sequence:

        if overall_stitched_image is None:
            overall_stitched_image = prepared_data[source_image_idx].original_page_img

        edge = stitching_sequence_graph.get_edge_data(source_image_idx, target_image_idx)
        first_idx = edge['first_idx']
        target_H = edge['H2']

        if source_image_idx in added_pages_transforms:
            source_H = added_pages_transforms[source_image_idx]
        else:
            source_H = np.eye(2, 3)
            added_pages_transforms[source_image_idx] = source_H

        if first_idx == target_image_idx:

            # build inverse affine transformation matrix
            extended_H = np.zeros((3, 3))
            extended_H[0:2, :] = target_H
            extended_H[2, 2] = 1

            inv = np.linalg.inv(extended_H)
            target_to_source_H = inv[0:2, :]

        else:
            target_to_source_H = target_H

        # TODO: actually we do not need overall_stitched_image pixels, and do not need blending inside stitch_next_image
        # we need only the region!
        overall_stitched_image, overall_H, updated_target_H = stitch_next_image(prepared_data=prepared_data,
                                                                                overall_image=overall_stitched_image,
                                                                                source_H=source_H,
                                                                                target_image_idx=target_image_idx,
                                                                                target_to_source_H=target_to_source_H)

        # update transformation matrices
        added_pages_transforms[target_image_idx] = updated_target_H

        # modify all already added pages transformations
        for k in added_pages_transforms.keys():
            if k != target_image_idx:
                ext_H1 = np.eye(3)
                ext_H1[0:2, :] = added_pages_transforms[k]

                ext_H2 = np.eye(3)
                ext_H2[0:2, :] = overall_H

                res = np.matmul(ext_H1, ext_H2)

                added_pages_transforms[k] = res[0:2, :]

    overall_region = [0, 0, overall_stitched_image.shape[1], overall_stitched_image.shape[0]]
    result_data = []

    for i in range(len(prepared_data)):
        h, w = prepared_data[i].original_page_img.shape[:2]

        cur_region = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        extended_H = np.zeros((3, 3))
        if not (i in added_pages_transforms):
            result_data.append({'image_rect': [], 'crop_region': []})
            raise HTTPException(500, f'Did not found any pair for page with index {i}')

        extended_H[0:2, :] = added_pages_transforms[i]
        extended_H[2, 2] = 1

        transformed_region = cv2.transform(np.array([cur_region]), extended_H)[0]
        image_rect = [int(v) for v in [transformed_region[0][0], transformed_region[0][1], transformed_region[2][0],
                                       transformed_region[2][1]]]

        result_data.append({'image_rect': image_rect, 'crop_region': [int(v) for v in prepared_data[i].crop_region]
        if prepared_data[i].crop_region is not None else []})

    return overall_region, result_data


def find_stitching_transformations(processed_images,
                                   placement_dict,
                                   pairwise_transformations_by_grids,
                                   stripe_width_percent,
                                   minimum_matches_count,
                                   minimum_inliers_count,
                                   inliers_threshold,
                                   maximum_rotation,
                                   matches_against_median_threshold,
                                   intersection_area_threshold,
                                   try_full_page,
                                   min_intersection_len_threshold,
                                   lowes_thr,
                                   max_keypoint_match_score):

    good_matches, match_counts = find_image_matches(processed_images, placement_dict=placement_dict,
                                                    stripe_width_percent=stripe_width_percent,
                                                    minimum_matches_count=minimum_matches_count,
                                                    minimum_inliers_count=minimum_inliers_count,
                                                    inliers_threshold=inliers_threshold,
                                                    maximum_rotation=maximum_rotation, try_full_page=try_full_page,
                                                    lowes_thr=lowes_thr,
                                                    intersection_area_threshold = intersection_area_threshold,
                                                    min_intersection_len_threshold = min_intersection_len_threshold
                                                    )

    pairwise_transformations_by_matcher = calculate_pairwise_transformations(
        good_matches=good_matches,
        match_counts=match_counts,
        matches_against_median_threshold=matches_against_median_threshold,
        inliers_threshold=inliers_threshold,
        stripe_width_percent=stripe_width_percent)

    if pairwise_transformations_by_matcher is not None and len(pairwise_transformations_by_matcher) > 0:
        pairwise_transformations = []
        if pairwise_transformations_by_grids is not None and len(pairwise_transformations_by_grids) > 0:
            for pair in pairwise_transformations_by_grids:
                score = pairwise_transformations_by_grids[pair][
                            1] + max_keypoint_match_score  # add max score to ensure that grid-based transformations are preferred
                pairwise_transformations.append(
                    (pair[1], pair[0], pairwise_transformations_by_grids[pair][0], -1, -1, score))
                # i, j, H, img1_stripe_id, img2_stripe_id, match_score

            for pair in pairwise_transformations_by_matcher:
                if (pair[0], pair[1]) not in pairwise_transformations_by_grids:
                    transformation_data = [*pair]
                    pairwise_transformations.append(transformation_data)
        else:
            pairwise_transformations = pairwise_transformations_by_matcher

        overall_region, transformations = stitch_images_overall(pairwise_transformations=pairwise_transformations,
                                                                prepared_data=processed_images)
        return overall_region, transformations
    else:
        return [], []


def list_of_pair_point(dct1, dct2):
    # point = ('A','B')
    lst1 = []
    lst2 = []
    lstnames = []
    for point in dct1:
        res1 = dct1.get(point, None)
        res2 = dct2.get(point, None)
        if not res2:
            res2 = dct2.get(point[::-1])
        if not res2:
            continue
        lst1.append(res1)
        lst2.append(res2)
        lstnames.append(point)
    return lst1, lst2, lstnames


def get_matrix(lst0, lst1):
    src_pts = np.array(lst0).astype(np.float32)
    dst_pts = np.array(lst1).astype(np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if len(lst0) <= 2 or len(lst1) <= 2:
        return 0

    matrix[0, 0] = round(matrix[0, 0])
    matrix[1, 1] = round(matrix[1, 1])
    matrix[0, 1] = round(matrix[0, 1])
    matrix[1, 0] = round(matrix[1, 0])

    return matrix


def get_transformation_for_pair_by_grids(dct_grid_page0, dct_grid_page1, BBox_segment_page0, BBox_segment_page1,
                                         img_size0,
                                         img_size1):
    intersection_dct0 = get_points_of_intersection_scale(dct_grid_page0, BBox_segment_page0, img_size0)
    intersection_dct1 = get_points_of_intersection_scale(dct_grid_page1, BBox_segment_page1, img_size1)

    H = None
    H_inv = None

    lst_pair0, lst_pair1, point_names = list_of_pair_point(intersection_dct0, intersection_dct1)
    matched_points_count = min(len(lst_pair0), len(lst_pair1))

    if matched_points_count > 2:
        H = get_matrix(lst_pair0, lst_pair1)
        H_inv = get_matrix(lst_pair1, lst_pair0)
    # matrixes.append([get_matrix(lst_pair0[i], lst_pair1[i], type_of_matrixes) for  i in range(len(lst_pair0))])

    return H, H_inv, matched_points_count


def get_pairwise_transformations_by_grids(crop_regions_for_pages, grids_for_pages, page_sizes, *,
                                          min_intersection_len_threshold=0.3, intersection_area_threshold=0.7,
                                          scale_threshold=0.001, rotation_threshold=0.001):
    pairwise_matching_info = {}

    if grids_for_pages is None:
        return pairwise_matching_info

    for i1 in range(len(crop_regions_for_pages)):
        for i2 in range(i1 + 1, len(crop_regions_for_pages)):
            if grids_for_pages[i1] is None or grids_for_pages[i2] is None:
                continue

            try:
                H, H_inv, matched_points_count = get_transformation_for_pair_by_grids(grids_for_pages[i1],
                                                                                      grids_for_pages[i2],
                                                                                      crop_regions_for_pages[i1],
                                                                                      crop_regions_for_pages[i2],
                                                                                      page_sizes[i1],
                                                                                      page_sizes[i2])
                if H is not None:
                    score = matched_points_count
                    scale = np.sqrt(H[0][0] * H[0][0] + H[0][1] * H[0][1])
                    rot = np.arctan2(H[1][0], H[0][0]) * 180 / np.pi

                    if abs(1 - scale) > scale_threshold:
                        continue

                    if abs(rot) > rotation_threshold and abs(abs(rot) - math.pi / 2) > rotation_threshold:
                        continue

                    intersect_area_proportion, intersect_side_proportion = get_page_rects_intersection_info(
                        abs(crop_regions_for_pages[i1][2] - crop_regions_for_pages[i1][0]),
                        abs(crop_regions_for_pages[i1][3] - crop_regions_for_pages[i1][1]),
                        abs(crop_regions_for_pages[i2][2] - crop_regions_for_pages[i2][0]),
                        abs(crop_regions_for_pages[i2][3] - crop_regions_for_pages[i2][1]), H)

                    if intersect_area_proportion > intersection_area_threshold:
                        score *= 0.1

                    if intersect_side_proportion < min_intersection_len_threshold:
                        score *= 0.1

                    pairwise_matching_info[(i1, i2)] = (H, score)
                    pairwise_matching_info[(i2, i1)] = (H_inv, score)
            except Exception as e:
                raise e

    return pairwise_matching_info


def get_stitching_sequence(pairwise_transformations: list[tuple], *, max_score=10000.0):
    stitching_sequence_graph = nx.Graph()
    max_page_index = -1
    for (i, j) in pairwise_transformations.keys():
        H, score = pairwise_transformations[(i, j)]
        stitching_sequence_graph.add_edge(i, j, first_idx=i, second_idx=j, H=H,
                                          weight=1.0 / score if score > 1.0 / max_score else max_score)
        max_page_index = max(i, j, max_page_index)

    stitching_sequence_graph = nx.minimum_spanning_tree(stitching_sequence_graph, weight='weight')

    start_page_idx = 0
    # check if start_page_idx is in the graph
    while start_page_idx not in stitching_sequence_graph and start_page_idx <= max_page_index:
        start_page_idx += 1

    if start_page_idx > max_page_index:
        return None, None

    stitch_sequence = list(nx.bfs_edges(stitching_sequence_graph, start_page_idx))

    return stitch_sequence


def calculate_final_transformations(stitch_sequence: list[tuple], pairwise_transformations: list[tuple],
                                    crop_regions: list[tuple[int, int, int, int]]) -> dict[int: np.ndarray]:
    resulting_transforms = {}
    stitch_sequence_list = []

    first_page_idx = stitch_sequence[0][0]
    resulting_transforms[first_page_idx] = np.eye(2, 3)
    stitch_sequence_list.append(first_page_idx)

    for p1_idx, p2_idx in stitch_sequence:
        if (p2_idx, p1_idx) in pairwise_transformations:
            H, _ = pairwise_transformations[(p2_idx, p1_idx)]
            H = H[:, :]
        elif (p1_idx, p2_idx) in pairwise_transformations:
            direct_H, _ = pairwise_transformations[(p1_idx, p2_idx)]
            extended_H = np.zeros((3, 3))
            extended_H[0:2, :] = direct_H
            extended_H[2, 2] = 1

            try:
                inv = np.linalg.inv(extended_H)
                H = inv[0:2, :]
            except:
                continue
        else:
            continue

        if p1_idx not in resulting_transforms:
            resulting_transforms[p1_idx] = np.eye(2, 3)

        source_H = resulting_transforms[p1_idx]

        # H is 2x3, we need to convert it to 3x3
        H1 = np.vstack([source_H, [0, 0, 1]])
        H2 = np.vstack([H, [0, 0, 1]])

        combined_H = np.matmul(H2, H1)
        # now strip back the last row, as we do not need it
        combined_H = combined_H[:2, :]

        page_source_region = crop_regions[p2_idx]
        w = page_source_region[2] - page_source_region[0]
        h = page_source_region[3] - page_source_region[1]

        image_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        transformed_region = cv2.transform(np.array([image_rect]), combined_H)[0]

        sx = 0
        sy = 0
        if transformed_region[0][0] < 0:
            sx = -transformed_region[0][0]
        if transformed_region[0][1] < 0:
            sy = -transformed_region[0][1]

        # so, we need to shift the whole image by sx, sy - that means we need to add sx, sy
        # to the transformation matrix for every image rectangle placed to overall so far
        if sx > 0 or sy > 0:
            shift_H = np.array([
                [1, 0, sx],
                [0, 1, sy],
                [0, 0, 1]
            ])
            for p in resulting_transforms:
                resulting_transforms[p] = np.matmul(shift_H, np.vstack([resulting_transforms[p], [0, 0, 1]]))
                resulting_transforms[p] = resulting_transforms[p][:2, :]  # make it 2x3 again

            H_mod = np.vstack([combined_H, [0, 0, 1]])
            combined_H = np.matmul(shift_H, H_mod)
            combined_H = combined_H[:2, :]

        resulting_transforms[p2_idx] = combined_H
        stitch_sequence_list.append(p2_idx)

    return resulting_transforms, stitch_sequence_list

def get_final_placements_for_pages(final_transforms: dict[int: np.ndarray],
                                   page_crop_regions: list[tuple[int, int, int, int]], *, use_rounding=False) -> (
        tuple)[dict[int:tuple[int, int, int, int]], int, int]:
    resulting_regions = {}
    for page_idx in final_transforms:
        H = final_transforms[page_idx]

        # TODO: think - what if we take non-cropped page region? May be it will be more interesting with good overlap slicing logic...
        page_source_region = page_crop_regions[page_idx]

        w = page_source_region[2] - page_source_region[0]
        h = page_source_region[3] - page_source_region[1]

        image_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        transformed_region = cv2.transform(np.array([image_rect]), H)[0]

        # use int, round or other logic? Now we have shift only, so at least dimensions will be the same...
        if use_rounding:
            resulting_regions[page_idx] = [round(transformed_region[0][0]), round(transformed_region[0][1]),
                                           round(transformed_region[2][0]), round(transformed_region[2][1])]
        else:
            resulting_regions[page_idx] = [int(transformed_region[0][0]), int(transformed_region[0][1]),
                                           int(transformed_region[2][0]), int(transformed_region[2][1])]

    resulting_w = 0
    resulting_h = 0
    for page_idx in resulting_regions:
        r = resulting_regions[page_idx]
        resulting_w = max(resulting_w, r[2])
        resulting_h = max(resulting_h, r[3])

    return resulting_regions, resulting_w, resulting_h


