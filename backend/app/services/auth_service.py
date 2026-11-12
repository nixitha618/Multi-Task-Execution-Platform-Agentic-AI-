import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from sqlalchemy import select, or_
from backend.app.core.datetime_utils import get_ist_now

from backend.app.core.database import AsyncSessionLocal
from backend.app.models.db_models import User
from backend.app.core.security import (
    validate_email_format,
    validate_password_strength,
    hash_password,
    verify_password,
    generate_session_token
)

logger = logging.getLogger(__name__)

class AuthService:
    """Authentication and User Account Service managing SQLite persistence."""

    async def register_user(
        self,
        email: str,
        password: str,
        confirm_password: Optional[str] = None,
        username: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Registers a new user with strong validation constraints and bcrypt password hashing.
        """
        # Confirm password check if provided
        if confirm_password is not None and confirm_password != password:
            return False, None, "Password confirmation does not match password."

        # 1. Email format validation
        valid_email, email_res = validate_email_format(email)
        if not valid_email:
            return False, None, email_res
        clean_email = email_res

        # 2. Password strength validation
        valid_pwd, pwd_errors = validate_password_strength(password)
        if not valid_pwd:
            return False, None, " • ".join(pwd_errors)

        # 3. Clean or generate username
        if username and str(username).strip():
            clean_username = str(username).strip().lower()
            if len(clean_username) < 3 or len(clean_username) > 50:
                return False, None, "Username must be between 3 and 50 characters."
        else:
            clean_username = clean_email.split("@")[0][:40]

        clean_name = str(full_name).strip() if full_name else None

        async with AsyncSessionLocal() as session:
            # Check existing email
            stmt_email = select(User).where(User.email == clean_email)
            res_email = await session.execute(stmt_email)
            existing_email = res_email.scalar_one_or_none()
            if existing_email:
                return False, None, f"An account with email '{clean_email}' already exists. Please log in."

            # Check existing username
            stmt_user = select(User).where(User.username == clean_username)
            res_user = await session.execute(stmt_user)
            existing_user = res_user.scalar_one_or_none()
            if existing_user:
                import uuid
                clean_username = f"{clean_username}_{uuid.uuid4().hex[:4]}"

            # 4. Hash password with bcrypt
            pwd_hash = hash_password(password)

            # 5. Create user record
            new_user = User(
                username=clean_username,
                email=clean_email,
                password_hash=pwd_hash,
                full_name=clean_name,
                is_active=True,
                last_login=get_ist_now()
            )
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)

            user_data = {
                "id": new_user.id,
                "username": new_user.username,
                "email": new_user.email,
                "full_name": new_user.full_name,
                "created_at": new_user.created_at,
                "last_login": new_user.last_login
            }
            logger.info(f"Successfully registered new user: {clean_email} (ID: {new_user.id})")
            return True, user_data, None

    async def authenticate_user(
        self,
        login_identifier: str,
        password: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Authenticates a user against SQLite stored credentials using bcrypt verification.
        Supports login with either email or username.
        """
        clean_id = str(login_identifier).strip().lower()
        if not clean_id or not password:
            return False, None, "Please provide both an email/username and a password."

        async with AsyncSessionLocal() as session:
            # Query by email or username
            stmt = select(User).where(
                or_(User.email == clean_id, User.username == clean_id)
            )
            res = await session.execute(stmt)
            user = res.scalar_one_or_none()

            if not user:
                return False, None, "Invalid email/username or password."

            if not user.password_hash:
                return False, None, "Password not configured for this account. Please register with a new password."

            # Verify bcrypt hash
            if not verify_password(password, user.password_hash):
                return False, None, "Invalid email/username or password."

            if not user.is_active:
                return False, None, "This user account is inactive. Please contact administrator."

            # Update last_login
            user.last_login = get_ist_now()
            await session.commit()
            await session.refresh(user)

            user_data = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at,
                "last_login": user.last_login
            }
            logger.info(f"User authenticated successfully: {user.email}")
            return True, user_data, None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(User).where(User.id == user_id)
            res = await session.execute(stmt)
            user = res.scalar_one_or_none()
            if not user:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at,
                "last_login": user.last_login
            }

auth_service = AuthService()