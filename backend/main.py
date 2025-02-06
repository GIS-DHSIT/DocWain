from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from backend.routes import auth, admin, documents
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Load templates from the "templates" directory
templates = Jinja2Templates(directory="templates")

# ✅ Include API Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.mount("/backend/static", StaticFiles(directory="static"), name="static")

# ✅ Render Landing Page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Render Login Page
@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# ✅ Render Signup Page
@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

# ✅ Render Chatbot Page
@app.get("/chatbot")
def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# ✅ Render Admin Page
@app.get("/admin")
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# ✅ Render Document Viewer Page
@app.get("/documents")
def document_page(request: Request):
    return templates.TemplateResponse("documents.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
