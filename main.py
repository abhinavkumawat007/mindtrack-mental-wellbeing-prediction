"""MindTrack — Root Application Wrapper.

Allows starting the server either via:
    uvicorn main:app --reload
or
    uvicorn backend.app.main:app --reload
"""

from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)