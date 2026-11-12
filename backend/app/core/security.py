import re
import secrets
import bcrypt
from typing import Tuple, List, Any

PASSWORD_REGEX = {
    "min_length": re.compile(r"^.{8,}$"),
    "uppercase": re.compile(r"[A-Z]"),
    "lowercase": re.compile(r"[a-z]"),
    "digit": re.compile(r"[0-9]"),
    "special": re.compile(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/`~]")
}

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

def validate_email_format(email: str) -> Tuple[bool, str]:
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    clean = email.strip()
    if len(clean) < 5 or len(clean) > 255:
        return False, "Email must be between 5 and 255 characters."
    if not EMAIL_REGEX.match(clean):
        return False, "Invalid email address format (e.g., user@example.com)."
    return True, clean.lower()

def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validates password against strong security constraints:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    - At least 1 special symbol
    """
    errors = []
    if not password or not isinstance(password, str):
        return False, ["Password is required."]
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if len(password) > 128:
        errors.append("Password cannot exceed 128 characters.")
    if not PASSWORD_REGEX["uppercase"].search(password):
        errors.append("Password must contain at least one uppercase letter (A-Z).")
    if not PASSWORD_REGEX["lowercase"].search(password):
        errors.append("Password must contain at least one lowercase letter (a-z).")
    if not PASSWORD_REGEX["digit"].search(password):
        errors.append("Password must contain at least one number (0-9).")
    if not PASSWORD_REGEX["special"].search(password):
        errors.append("Password must contain at least one special character (!@#$%^&* etc.).")
        
    return len(errors) == 0, errors

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with automatic salting."""
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def generate_session_token(user_id_or_length: Any = 32, length: int = 32) -> str:
    """Generates a cryptographically secure random session token."""
    nbytes = length
    if isinstance(user_id_or_length, int):
        nbytes = user_id_or_length
    return secrets.token_hex(nbytes)