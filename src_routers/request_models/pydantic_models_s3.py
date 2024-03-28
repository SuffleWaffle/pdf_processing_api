# >>>> </> STANDARD IMPORTS </>
# >>>> ********************************************************************************
from typing import Optional, List
# >>>> ********************************************************************************

# >>>> </> EXTERNAL IMPORTS </>
# >>>> ********************************************************************************
from pydantic import BaseModel
# >>>> ********************************************************************************


class PDFFileDetails(BaseModel):
    file_key: str = "SAMPLE_FILE_NAME.pdf"


class JSONFileDetails(BaseModel):
    file_key: str = "SAMPLE_FILE_NAME.json"


class PDFStitchingNoOverallFiles(BaseModel):
    pdfs_to_stitch: Optional[List[str]] = ["SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf"]
    match_lines: Optional[List[str]] = ["SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json"]
    segmentation_info: Optional[List[str]] = ["SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json"]
    parsed_text: Optional[List[str]] = ["SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json"]
    grid_lines: Optional[List[str]] = ["SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json"]
    remove_text: Optional[bool] = False

class PDFStitchingNoOverallFilesDataS3(BaseModel):
    files: PDFStitchingNoOverallFiles
    s3_bucket_name: str = "drawer-ai-services-test-files"
    out_s3_file_key: str = "SAMPLE_FILE_NAME.pdf"
    remove_text: Optional[bool] = False

    class Config:
        schema_extra = {
            "files": {
                "pdfs_to_stitch": [
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf"
                ],
                "match_lines" : ["SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json"],
                "segmentation_info": [
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json"
                ],
                "grid_lines": [
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json"
                ],
                "parsed_text" : ["SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json"]
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
            "out_s3_file_key": "SAMPLE_FILE_NAME.pdf",
            "remove_text": False
        }
class PDFStitchingWithOverallFiles(BaseModel):
    overall_plan_pdf: PDFFileDetails
    pdfs_to_stitch: List[str] = ["SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf", "SAMPLE_FILE_NAME.pdf"]
    overall_plan_grids_json: JSONFileDetails
    grids_to_stitch_jsons: List[str] = ["SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json", "SAMPLE_FILE_NAME.json"]


class PDFStitchingWithOverallFilesDataS3(BaseModel):
    files: PDFStitchingWithOverallFiles
    s3_bucket_name: str = "drawer-ai-services-test-files"
    out_s3_file_key: str = "SAMPLE_FILE_NAME.pdf"
    class Config:
        schema_extra = {
            "files": {
                "overall_plan_pdf": {
                    "file_key": "SAMPLE_FILE_NAME.pdf"
                },
                "pdfs_to_stitch": [
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf",
                    "SAMPLE_FILE_NAME.pdf"
                ],
                "overall_plan_grids_json": {
                    "file_key": "SAMPLE_FILE_NAME.json"
                },
                "grids_to_stitch_jsons": [
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json",
                    "SAMPLE_FILE_NAME.json"
                ]
            },
            "s3_bucket_name": "drawer-ai-services-test-files",
            "out_s3_file_key": "SAMPLE_FILE_NAME.pdf"
        }
