from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.compare import router as compare_router
from routes.health import router as health_router

app = FastAPI(title="Age-Invariant Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compare_router)
app.include_router(health_router)


@app.get("/")
async def root():
    return {"message": "Age-Invariant Face Recognition API"}
