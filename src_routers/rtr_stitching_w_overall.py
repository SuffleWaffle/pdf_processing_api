# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
import io
from typing import List
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, status, Response
import ujson as json
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
# ---- CONFIG ----
import settings
from src_logging import log_config
# ---- PROCESSES ----
from src_processes.stitch_pdf_w_overall import stitch_pdf_w_overall
# ---- UTILS ----
from src_utils.loading_utils import load_pdf
from src_utils.aws_utils import S3Utils
# ---- REQUEST MODELS ----
from src_routers.request_models.pydantic_models_s3 import PDFStitchingWithOverallFilesDataS3
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

# ________________________________________________________________________________
# --- FastAPI ROUTER ---
pdf_stitching_pdf_rtr_v1 = APIRouter(prefix="/v1")


# ________________________________________________________________________________
@pdf_stitching_pdf_rtr_v1.post(path="/pdf-stitching-pdf/",
                               status_code=status.HTTP_200_OK,
                               responses={200: {}, 500: {}, 503: {}},
                               response_class=Response,
                               response_description="Stitch PDF - WITH Overall",
                               tags=['PDF Stitching - WITH Overall'],
                               summary="Stitch PDFs (WITH overall) and return resulting PDF file")
async def pdf_stitching_pdf(overall_plan_pdf: UploadFile = File(...),
                            pdfs_to_stitch: List[UploadFile] = File(...),
                            overall_plan_grids: UploadFile = File(...),
                            grids_to_stitch: List[UploadFile] = File(...)):
    # load overall plan grids
    if not overall_plan_grids.filename.endswith('.json'):
        raise HTTPException(404, f'File with filename {overall_plan_grids.filename} does not end with .json')

    overall_plan_grids = json.loads(await overall_plan_grids.read())

    # load overall plan pdf
    if not overall_plan_pdf.filename.endswith('.pdf'):
        raise HTTPException(404, f'File with filename {overall_plan_pdf.filename} does not end with .pdf')
    overall_plan_doc = load_pdf(overall_plan_pdf, 0)[0]

    # load grids to stitch
    grids_to_stitch_res = []
    for file in grids_to_stitch:
        if not file.filename.endswith('.json'):
            raise HTTPException(404, f'File with filename {file.filename} does not end with .json')

        grids_to_stitch_res.append(json.loads(await file.read()))

    # load pdfs to stitch
    pdfs_to_stitch_res = []
    for file in pdfs_to_stitch:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(404, f'File with filename {file.filename} does not end with .pdf')

        pdfs_to_stitch_res.append(load_pdf(file, 0)[0])

    # actual process
    stitched = stitch_pdf_w_overall(overall_plan_doc=overall_plan_doc,
                                    overall_plan_grids=overall_plan_grids,
                                    grids_list=grids_to_stitch_res,
                                    docs_list=pdfs_to_stitch_res,
                                    config=settings.STITCH_PDF)

    return Response(content=stitched,
                    media_type='application/pdf',
                    status_code=status.HTTP_200_OK,
                    headers={'Content-Disposition': 'inline; filename="stitched.pdf"'})


# ________________________________________________________________________________
@pdf_stitching_pdf_rtr_v1.post(path="/pdf-stitching-pdf-s3/",
                               status_code=status.HTTP_200_OK,
                               responses={200: {}, 500: {}, 503: {}},
                               response_class=Response,
                               response_description="Stitch PDF - With Overall",
                               tags=['A - Production', 'PDF Stitching - With Overall', 'S3'],
                               summary="Stitch PDFs (WITH overall) and upload resulting PDF file to S3")
async def pdf_stitching_s3(files_data: PDFStitchingWithOverallFilesDataS3) -> Response:
    # ________________________________________________________________________________
    # --- INIT S3 UTILS INSTANCE ---
    s3 = S3Utils()

    # --- DOWNLOAD | OVERALL PLAN PDF | FROM AWS S3 ---
    overall_plan_pdf_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                  s3_file_key=files_data.files.overall_plan_pdf.file_key)
    overall_plan_pdf = load_pdf(pdf_file_obj=overall_plan_pdf_bytes,
                                page_num=0,
                                s3_origin=True)[0]

    # --- DOWNLOAD | PDFs TO STITCH | LIST OF PDFs FROM AWS S3 ---
    pdfs_to_stitch_res = []
    for filey_key in files_data.files.pdfs_to_stitch:
        pdf_to_stitch_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                   s3_file_key=filey_key)
        pdfs_to_stitch_res.append(load_pdf(pdf_file_obj=pdf_to_stitch_bytes,
                                           page_num=0,
                                           s3_origin=True)[0])

    # --- DOWNLOAD | OVERALL PLAN GRIDS | JSON FROM AWS S3 ---
    overall_plan_grids_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                         s3_file_key=files_data.files.overall_plan_grids_json.file_key)
    overall_plan_grids_json_decoded = overall_plan_grids_json_bytes.getvalue().decode("utf-8")
    overall_plan_grids = json.loads(overall_plan_grids_json_decoded)

    # --- DOWNLOAD | GRIDS TO STITCH | LIST OF JSONs FROM AWS S3 ---
    grids_to_stitch_res = []
    for filey_key in files_data.files.grids_to_stitch_jsons:
        grid_to_stitch_json_bytes = s3.download_file_obj(s3_bucket_name=files_data.s3_bucket_name,
                                                         s3_file_key=filey_key)
        grid_to_stitch_json_decoded = grid_to_stitch_json_bytes.getvalue().decode("utf-8")
        grids_to_stitch_res.append(json.loads(grid_to_stitch_json_decoded))

    # ________________________________________________________________________________
    # - ACTUAL PDF STITCHING PROCESS
    # ________________________________________________________________________________
    stitched_pdf_fitz_file_obj = stitch_pdf_w_overall(overall_plan_doc=overall_plan_pdf,
                                            overall_plan_grids=overall_plan_grids,
                                            grids_list=grids_to_stitch_res,
                                            docs_list=pdfs_to_stitch_res,
                                            config=settings.STITCH_PDF)

    # ________________________________________________________________________________
    # --- CONVERT FITZ PDF FILE OBJECT TO BytesIO STREAM ---
    logger.info(f"- Converting stitched PDF file object to BytesIO stream.")
    logger.info(f"- Type of file object befor conversion: {type(stitched_pdf_fitz_file_obj)}")
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
