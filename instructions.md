How to install all the requirements:
pip install -r requirements.txt

Command for Inference file:
python scripts/run_inference.py --dataset medqa --method mog --llm llama3

How to run the app:
pip install fastapi uvicorn
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload