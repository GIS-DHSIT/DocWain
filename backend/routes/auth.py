from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt import JWT, supported_key_types
import datetime
import bcrypt
from database.duckdb_handler import get_user, create_user

router = APIRouter()

SECRET_KEY = "dhsitDocWain".encode("utf-8")
ALGORITHM = "HS256"

# Hash Password with bcrypt
def hash_password(password: str):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

# Verify Password
# def verify_password(plain_password, hashed_password):
#     return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
#
# # Generate JWT Token
# def create_access_token(user_id: str):
#     expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
#     return jwt.encode({"sub": user_id, "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# Generate JWT Token
def create_access_token(user_id: str):
    expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    expiration_unix = int(expiration.timestamp())
    key = supported_key_types()['oct'](SECRET_KEY)
    return JWT().encode({"sub": user_id, "exp": expiration_unix}, key, alg=ALGORITHM)

# User Signup API (Uses bcrypt)
@router.post("/signup")
def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...), domain: str = Form(...)):
    if get_user(email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd = hash_password(password)  # Use bcrypt to hash password
    user_id = create_user(username, email, hashed_pwd, domain)
    return {"message": "User registered successfully", "user_id": user_id}

# User Login API (Uses bcrypt)
# @router.post("/login")
# def login(email: str = Form(...), password: str = Form(...)):
#     user = get_user(email)
#     print(user)
#     if not user or not verify_password(password, user["password"]):
#         raise HTTPException(status_code=400, detail="Invalid credentials")
#
#     token = create_access_token(user["id"])
#     return {"access_token": token, "token_type": "bearer", "user": user}

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token(user["id"])
    return {"access_token": token, "token_type": "bearer", "user": user}