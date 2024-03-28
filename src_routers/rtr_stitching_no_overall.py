# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import io
import logging
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, status, Response

import settings
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from src_logging import log_config
from src_utils.loading_utils import load_pdf
from src_utils.aws_utils import S3Utils
from src_processes.stitch_pdf_no_overall import stitch_pdf_no_overall
from typing import List, Optional
import ujson as json
# ---- REQUEST MODELS ----
from src_routers.request_models.pydantic_models_s3 import PDFStitchingNoOverallFilesDataS3
# >>>> ********************************************************************************

# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

pdf_stitching_pdf_rtr_v2 = APIRouter(prefix="/v2")


@pdf_stitching_pdf_rtr_v2.post(path="/pdf-stitching-pdf/",
                               status_code=status.HTTP_200_OK,
                               responses={200: {}, 500: {}, 503: {}},
                               response_class=Response,
                               response_description="Stitch PDF - NO Overall",
                               tags=['PDF Stitching - NO Overall'],
                               summary="Stitch PDFs and and return resulting PDF file")
async def pdf_stitching_pdf(pdfs_to_stitch: List[UploadFile] = File(...),
                            match_lines: Optional[List[UploadFile]] | None = None,
                            segmentation_info: Optional[List[UploadFile]] | None = None,
                            parsed_text: Optional[List[UploadFile]] | None = None,
                            remove_text: bool = False,
                            grid_lines: Optional[List[UploadFile]] | None = None
                            ) -> Response:
    # load grids to stitch
    match_lines_res = None
    if match_lines:
        match_lines_res = []
        for file in match_lines:
            if not file.filename.endswith('.json'):
                raise HTTPException(404, f'File with filename {file.filename} does not end with .json')

            match_lines_res.append(json.loads(await file.read()))
        # logger.info(match_lines_res)

    # load segmentation info
    segmentation_info_res = None
    if segmentation_info:
        segmentation_info_res = []
        for file in segmentation_info:
            if not file.filename.endswith('.json'):
                raise HTTPException(404, f'File with filename {file.filename} does not end with .json')

            segmentation_info_res.append(json.loads(await file.read()))
    # logger.info(segmentation_info_res)

    # load grid lines
    grid_lines_res = None
    if grid_lines:
        grid_lines_res = []
        for file in grid_lines:
            if file.filename.endswith('.json'):
                try:
                    grid_lines_res.append(json.loads(await file.read()))
                except Exception as e:
                    grid_lines_res.append([])
    # logger.info("Grid lines loaded: %s", grid_lines_res)

    # load pdfs to stitch
    pdfs_to_stitch_res = []
    images_to_stitch = []
    for file in pdfs_to_stitch:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(404, f'File with filename {file.filename} does not end with .pdf')

        doc, page, img_array, pdf_size = load_pdf(file, 0,
                                                  s3_origin=False)
        images_to_stitch.append(img_array)
        pdfs_to_stitch_res.append(doc)

    # load text for stitching
    parsed_text_res = None
    if parsed_text:
        parsed_text_res = []
        for file in parsed_text:
            if not file.filename.endswith('.json'):
                raise HTTPException(404, f'File with filename {file.filename} does not end with .json')

            parsed_text_res.append(json.loads(await file.read()))
    # actual process
    stitched = stitch_pdf_no_overall(docs_list=pdfs_to_stitch_res,
                                     parsed_text_list=parsed_text_res,
                                     images_list=images_to_stitch,
                                     match_lines_list=match_lines_res,
                                     crop_regions_list=segmentation_info_res,
                                     grid_lines_list=grid_lines_res,
                                     config=settings.STITCH_PDF_NO_OVERALL,
                                     remove_text=remove_text)

    return Response(stitched, media_type='application/pdf',
                    status_code=status.HTTP_200_OK,
                    headers={'Content-Disposition': 'inline; filename="stitched.pdf"'})


@pdf_stitching_pdf_rtr_v2.post(path="/pdf-stitching-pdf-s3/",
                               status_code=status.HTTP_200_OK,
                               responses={200: {}, 500: {}, 503: {}},
                               response_class=Response,
                               response_description="Stitch PDF - No Overall",
                               tags=['A - Production', 'PDF Stitching - No Overall', 'S3'],
                               summary="Stitch PDFs (NO overall) and upload resulting PDF file to S3")
async def pdf_stitching_s3(files_data: PDFStitchingNoOverallFilesDataS3) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3 UTILS INSTANCE ---
    s3 = S3Utils()

    # --- DOWNLOAD | PDFs TO STITCH | LIST OF PDFs FROM AWS S3 ---
    pdfs_to_stitch_res = []
    images_to_stitch = []
    for filey_key in files_data.files.pdfs_to_stitch:
        pdf_to_stitch_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                   s3_file_key=filey_key)
        doc, page, img_array, pdf_size = load_pdf(pdf_file_obj=pdf_to_stitch_bytes,
                                                  page_num=0,
                                                  s3_origin=True)
        images_to_stitch.append(img_array)
        pdfs_to_stitch_res.append(doc)

    # --- DOWNLOAD | Segmentation info | LIST OF JSONs FROM AWS S3 ---
    segmentation_info_res = []
    for filey_key in files_data.files.segmentation_info:
        segmentation_info_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                            s3_file_key=filey_key)
        segmentation_info_json_decoded = segmentation_info_json_bytes.getvalue().decode("utf-8")
        segmentation_info_res.append(json.loads(segmentation_info_json_decoded))

    # --- DOWNLOAD | Grid lines | LIST OF JSONs FROM AWS S3 ---
    grid_lines_res = []
    for filey_key in files_data.files.grid_lines:
        grid_lines_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                            s3_file_key=filey_key)
        grid_lines_json_decoded = grid_lines_json_bytes.getvalue().decode("utf-8")
        grid_lines_res.append(json.loads(grid_lines_json_decoded))


    # --- DOWNLOAD | Match lines | LIST OF JSONs FROM AWS S3 ---
    match_lines_res = []
    for filey_key in files_data.files.match_lines:
        match_lines_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                      s3_file_key=filey_key)
        match_lines_json_decoded = match_lines_json_bytes.getvalue().decode("utf-8")
        match_lines_res.append(json.loads(match_lines_json_decoded))
    # --- DOWNLOAD | Parsed text | LIST OF JSONs FROM AWS S3 ---
    parsed_text_res = []
    for filey_key in files_data.files.parsed_text:
        parsed_text_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                      s3_file_key=filey_key)
        parsed_text_json_decoded = parsed_text_json_bytes.getvalue().decode("utf-8")
        parsed_text_res.append(json.loads(parsed_text_json_decoded))
    # ________________________________________________________________________________
    # - ACTUAL PDF STITCHING PROCESS
    # ________________________________________________________________________________
    stitched_pdf_fitz_file_obj = stitch_pdf_no_overall(docs_list=pdfs_to_stitch_res,
                                                       parsed_text_list=parsed_text_res,
                                                       images_list=images_to_stitch,
                                                       match_lines_list=match_lines_res,
                                                       crop_regions_list=segmentation_info_res,
                                                       grid_lines_list=grid_lines_res,
                                                       config=settings.STITCH_PDF_NO_OVERALL,
                                                       remove_text=files_data.remove_text)

    # ________________________________________________________________________________
    # --- CONVERT FITZ PDF FILE OBJECT TO BytesIO STREAM ---
    pdf_byte_stream = io.BytesIO(stitched_pdf_fitz_file_obj)

    # --- UPLOAD RESULT TO AWS S3 BUCKET ---
    try:
        s3_upload_status = s3.upload_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                              s3_file_key=files_data.out_s3_file_key,
                                              file_byte_stream=pdf_byte_stream)
        if not s3_upload_status:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"ERROR -> S3 upload status: {s3_upload_status}")
        return Response(status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"ERROR -> Failed to upload file to S3. Error: {e}")
