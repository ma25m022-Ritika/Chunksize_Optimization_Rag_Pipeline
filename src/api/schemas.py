from pydantic import BaseModel
from typing import Optional, Dict

class QuestionRequest(BaseModel):
    question: str
    options: Optional[Dict[str, str]] = None
    method: str = "mog"  # cot | medrag | mog
    top_k: int = 5

class AnswerResponse(BaseModel):
    question: str
    answer: str
    answer_line: str = ""
    prediction: str
    method: str
    retrieved_docs: list = []
    router_probs: Optional[Dict[str, float]] = None