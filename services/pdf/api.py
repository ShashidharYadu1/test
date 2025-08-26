import uvicorn
from fastapi import FastAPI, BackgroundTasks, status
from pydantic import BaseModel
from typing import Optional, List
from pipeline_pdf import run_pdf_pipeline_logic


class PdfProcessRequest(BaseModel):
    pdf_input_key: str
    page_no: Optional[List[int]] = None

app = FastAPI(
    title="PDF Processing Service",
    description="An API to trigger the PDF extraction, conversion, and processing pipeline."
)

@app.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_pdf_file(
    request: PdfProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Accepts a PDF processing request and starts the pipeline in the background.
    """
    background_tasks.add_task(
        run_pdf_pipeline_logic,
        pdf_input_key=request.pdf_input_key,
        page_no=request.page_no
    )
    
    return {"message": "PDF processing job started in the background."}