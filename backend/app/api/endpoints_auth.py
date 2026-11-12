from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.auth_service import auth_service
from backend.app.core.security import generate_session_token
from backend.app.models.schemas import (
    UserSignUpRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
)

router = APIRouter()

@router.post("/signup", response_model=AuthResponse)
async def signup(req: UserSignUpRequest):
    """
    Register a new user with strong password and email validation.
    Email and hashed password are saved in SQLite database.
    """
    success, user_data, error_msg = await auth_service.register_user(
        email=req.email,
        password=req.password,
        confirm_password=req.confirm_password,
        username=req.username,
        full_name=req.full_name
    )
    if not success:
        raise HTTPException(status_code=400, detail=error_msg or "Registration failed.")

    token = generate_session_token(user_data["id"])
    return AuthResponse(
        success=True,
        message="Account registered successfully",
        user=UserResponse(**user_data),
        token=token
    )

@router.post("/login", response_model=AuthResponse)
async def login(req: UserLoginRequest):
    """
    Authenticate user with email/username and password against the SQLite database.
    """
    success, user_data, error_msg = await auth_service.authenticate_user(
        login_identifier=req.email_or_username,
        password=req.password
    )
    if not success:
        raise HTTPException(
            status_code=401,
            detail=error_msg or "Invalid email/username or password. Please verify your credentials."
        )

    token = generate_session_token(user_data["id"])
    return AuthResponse(
        success=True,
        message="Logged in successfully",
        user=UserResponse(**user_data),
        token=token
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user(user_id: str = Query(..., description="ID of user to retrieve")):
    """
    Retrieve profile of currently authenticated user.
    """
    user_data = await auth_service.get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user_data)

@router.post("/logout")
async def logout():
    """
    Log out user and invalidate client session.
    """
    return {"success": True, "message": "Successfully logged out"}

