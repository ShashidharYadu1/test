import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from pipeline_excel import run_excel_pipeline_logic


class ExcelProcessRequest(BaseModel):
    input_key: str
    sheet_name: str = Field(..., pattern="^(budget|assumption|timeline)$")
    error_key: Optional[str] = None
    previous_code_key: Optional[str] = None

app = FastAPI(
    title="Excel Processing Service",
    description="An API to trigger the Excel analysis, generation, and execution pipeline."
)

@app.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_excel_file(
    request: ExcelProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Accepts an Excel processing request and starts the pipeline in the background.
    """
    background_tasks.add_task(
        run_excel_pipeline_logic,
        input_key=request.input_key,
        sheet_name=request.sheet_name,
        error_key=request.error_key,
        previous_code_key=request.previous_code_key
    )
    
    return {"message": "Excel processing job started in the background."}