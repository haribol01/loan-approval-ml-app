from fastapi import FastAPI
from pydantic import BaseModel
from main import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Loan Approval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanRequest(BaseModel):
    cibil_score: int
    income_annum: float
    loan_amount: float
    no_of_dependents: int
    loan_term: int

class LoanResponse(BaseModel):
    approved: int

@app.post("/predict", response_model=LoanResponse)
def predict_loan(req: LoanRequest):
    result = predict(
        req.cibil_score,
        req.income_annum,
        req.loan_amount,
        req.no_of_dependents,
        req.loan_term
    )
    return {"approved": result}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 