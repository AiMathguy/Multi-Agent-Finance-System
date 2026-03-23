from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="NeuralX Finance Engine")
app.include_router(router)