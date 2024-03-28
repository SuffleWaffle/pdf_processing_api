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
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, status, Response
import httpx
import pytest
import ujson as json
# import requests
# from requests.auth import HTTPBasicAuth
from pydantic import BaseModel, Field
from typing import List
# >>>> ********************************************************************************

# >>>> </> LOCAL IMPORTS </>
# >>>> ********************************************************************************
from main import app
from src_logging import log_config
from src_tests import unit_tests
import tests_settings
from src_tests.unit_tests import assertion_s3, base_url, dev_username, dev_password
# >>>> ********************************************************************************


# ________________________________________________________________________________
# --- INIT CONFIG - LOGGER SETUP ---
logger = log_config.setup_logger(logger_name=__name__, logging_level=logging.DEBUG)

client = TestClient(app)


# ________________________________________________________________________________
# >>>> </> TEST - PDF STITCHING ENDPOINT - S3 </>
def test_pdf_stitching_s3():
    test_data = tests_settings.TEST_PDF_STITCHING_S3
    link = base_url + test_data["link"]

    json_data = test_data["json_data"]
    json_payload = json.dumps(json_data)

    response = client.post(url=link,
                           content=json_payload)

    # ________________________________________________________________________________
    # --- CHECK RESPONSE STATUS CODE ---
    # assertion_s3(response,
    #              status_code=200)


# ________________________________________________________________________________
# >>>> </> TEST - HEALTHCHECK </>
def test_healthcheck_get():
    response = client.get("/healthcheck/")
    assert response.status_code == 200
    assert response.json() == {
        "healthcheck": "API Status 200"
    }
