# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
import logging
# TEMPORARY
import os
import io
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
import json
# import requests
# from requests.auth import HTTPBasicAuth
from pydantic import BaseModel, Field
from typing import List
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from main import app
from src_logging import log_config
from src_utils.aws_utils import S3FileOps
from src_tests import unit_tests
import tests_settings
# >>>> ********************************************************************************

dev_username = "drawer"
dev_password = "Y4AuMasf"
base_url = tests_settings.BASE_URL

# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

client = TestClient(app)


# ________________________________________________________________________________
# >>>> </> TEST - PDF STITCHING ENDPOINT - S3 - FILE NOT FOUND </>
def test_pdf_stitching_s3_file_not_found():
    test_data = tests_settings.TEST_PDF_STITCHING_S3
    link = base_url + test_data["link"]

    json_data = test_data["json_data"]
    json_data["files"]["overall_plan_pdf"]["file_key"] = "wrong_file.pdf"
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assertion_s3(response,
                 status_code=503)


# ________________________________________________________________________________
# >>>> </> TEST - PDF STITCHING ENDPOINT - S3 - BAD CREDENTIALS </>
@pytest.mark.skip(reason="Not implemented")
def test_pdf_stitching_s3_bad_credentials():
    test_data = tests_settings.TEST_PDF_STITCHING_S3
    link = base_url + test_data["link"]

    json_data = test_data["json_data"]
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assertion_s3(response,
                 status_code=503)


# ________________________________________________________________________________
# >>>> </> TEST - EXTRACT ALL SHAPES DATA - JSON </>
@pytest.mark.skip(reason="Test is not working...")
def test_pdf_stitching_pdf_local():
    test_data = tests_settings.TEST_PDF_STITCHING_JSON
    link = base_url + test_data["link"]

    overall_plan_pdf_data = download_file_s3(test_data["file_keys"]["overall_plan_pdf"],
                                             test_data["s3_bucket_name"])
    pdfs_to_stitch_data = [
        download_file_s3(file_key,
                         test_data["s3_bucket_name"])
        for file_key in test_data["file_keys"]["pdfs_to_stitch"]
    ]
    overall_plan_grids_json_data = download_file_s3(test_data["file_keys"]["overall_plan_grids_json"],
                                                    test_data["s3_bucket_name"])
    grids_to_stitch_jsons_data = [
        download_file_s3(file_key,
                         test_data["s3_bucket_name"])
        for file_key in test_data["file_keys"]["grids_to_stitch_jsons"]
    ]

    files = {
        "overall_plan_pdf": (overall_plan_pdf_data.name, overall_plan_pdf_data, "application/pdf"),
        "pdfs_to_stitch": [
            (pdf.name, pdf, "application/pdf") for pdf in pdfs_to_stitch_data
        ],
        "overall_plan_grids": (overall_plan_grids_json_data.name, overall_plan_grids_json_data, "application/json"),
        "grids_to_stitch": [
            (grids_json.name, grids_json, "application/json") for grids_json in grids_to_stitch_jsons_data
        ]
    }

    response = client.post(url=link,
                           files=files)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE + CONTENT TYPE ---
    assertion_json(response,
                   status_code=200)


def assertion_json(response,
                   status_code: int):
    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assert response.status_code == status_code
    logger.info(f"- RESPONSE STATUS CODE - {response.status_code}")

    # ________________________________________________________________________________
    # --- CHECK RESPONSE CONTENT TYPE ---
    assert response.headers["content-type"] == "application/json"
    content_type = response.headers["content-type"]
    logger.info(f"- RESPONSE CONTENT TYPE - {content_type}")


def assertion_s3(response,
                 status_code: int):
    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    assert response.status_code == status_code
    logger.info(f"- RESPONSE STATUS CODE - {response.status_code}")


def download_n_save_file_s3(file_key, s3_bucket_name):
    # _____________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    logger.info("- CONNECTING TO AWS S3 BUCKET -")
    s3 = S3FileOps(s3_bucket_name=s3_bucket_name)

    # --- DOWNLOAD FILE FROM S3 ---
    file_data = s3.download_file_obj(s3_bucket_name, file_key)

    # --- SAVING FILE ---
    logger.info("- 2 - SAVING FILE -")
    filename = file_key.split('/')[-1]

    with open(filename, "wb") as f:
        f.write(file_data.getvalue())


def download_file_s3(file_key, s3_bucket_name):
    # _____________________________________________________________________________
    # --- INIT S3FileOps INSTANCE ---
    logger.info("- CONNECTING TO AWS S3 BUCKET -")
    s3 = S3FileOps(s3_bucket_name=s3_bucket_name)

    # --- DOWNLOAD FILE FROM S3 ---
    file_data = s3.download_file_obj(s3_bucket_name, file_key)
    file_data.name = file_key.split('/')[-1]

    return file_data
