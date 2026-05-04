from fastapi import APIRouter, HTTPException
from src.api.schemas import QuestionRequest, AnswerResponse
from src.api.service import answer_question

router = APIRouter()

@router.post("/answer", response_model=AnswerResponse)
def answer(request: QuestionRequest):
    try:
        result = answer_question(
            question=request.question,
            options=request.options,
            method=request.method,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {"status": "ok"}