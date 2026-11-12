import asyncio
import sqlite3
import uuid
import os
import sys
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.app.core.security import (
    validate_email_format,
    validate_password_strength,
    hash_password,
    verify_password
)
from backend.app.core.database import init_db
from backend.app.services.auth_service import auth_service
from backend.app.api.endpoints_auth import router as auth_router
from fastapi import FastAPI
from fastapi.testclient import TestClient

test_app = FastAPI()
test_app.include_router(auth_router, prefix="/api/v1/auth")


async def run_tests():
    print("========================================")
    print("AGENTIC AI AUTH SYSTEM VERIFICATION")
    print("========================================")
    
    # 1. Database Init
    print("\n--- 1. Testing Database Migration & Initialization ---")
    await init_db()
    conn = sqlite3.connect("data/agentic_ai.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    cols = [row[1] for row in cursor.fetchall()]
    print(f"Users table columns: {cols}")
    assert "password_hash" in cols, "password_hash missing from users table!"
    assert "email" in cols, "email missing from users table!"
    assert "full_name" in cols, "full_name missing from users table!"
    print("✓ SQLite database schema verified with all auth columns.")

    # 2. Email Validation Rules
    print("\n--- 2. Testing Email Validation Rules ---")
    valid_emails = ["test@example.com", "user.name@rgukt.ac.in", "nixitha_123@domain.org"]
    invalid_emails = ["plainaddress", "missing@domain", "@nodomain.com", "user@.com", "spaces in@email.com"]
    for em in valid_emails:
        ok, msg = validate_email_format(em)
        assert ok, f"Expected {em} to be valid, got: {msg}"
    print(f"✓ All {len(valid_emails)} valid email patterns accepted.")
    for em in invalid_emails:
        ok, msg = validate_email_format(em)
        assert not ok, f"Expected {em} to be invalid"
    print(f"✓ All {len(invalid_emails)} invalid email patterns rejected.")

    # 3. Password Strength Rules
    print("\n--- 3. Testing Password Strength Rules ---")
    weak_cases = [
        ("short1!", "at least 8 characters"),
        ("nouppercase123!", "uppercase letter"),
        ("NOLOWERCASE123!", "lowercase letter"),
        ("NoDigitsHere!!", "number"),
        ("NoSpecialChar123", "special character"),
    ]
    for pwd, expected_err in weak_cases:
        ok, msg = validate_password_strength(pwd)
        assert not ok, f"Expected password '{pwd}' to be rejected"
        assert any(expected_err.lower() in m.lower() for m in msg), f"Expected '{expected_err}' in error message, got: {msg}"
        print(f"  ✓ Rejected weak password ({expected_err}): {msg}")

    strong_pwd = "SecureP@ssword2026!"
    ok, msg = validate_password_strength(strong_pwd)
    assert ok, f"Expected strong password to pass, got: {msg}"
    print(f"✓ Strong password accepted: '{strong_pwd}'")

    # 4. Password Hashing & Bcrypt 12 Rounds
    print("\n--- 4. Testing Bcrypt Salted Hashing ---")
    hashed = hash_password(strong_pwd)
    print(f"Hashed output: {hashed[:30]}...")
    assert hashed.startswith("$2b$12$"), f"Expected bcrypt 12-round format, got: {hashed[:7]}"
    assert verify_password(strong_pwd, hashed) is True, "Password verification failed for correct password!"
    assert verify_password("WrongPassword123!", hashed) is False, "Password verification erroneously succeeded for wrong password!"
    print("✓ Bcrypt hashing and salt verification verified.")

    # 5. AuthService Registration & SQLite Storage
    print("\n--- 5. Testing User Registration in SQLite DB ---")
    test_email = f"auth_test_{uuid.uuid4().hex[:6]}@rgukt.ac.in"
    reg_ok, user_data, err = await auth_service.register_user(
        email=test_email,
        password=strong_pwd,
        confirm_password=strong_pwd,
        username=f"user_{uuid.uuid4().hex[:4]}",
        full_name="Nixitha Unta"
    )
    assert reg_ok is True, f"Registration failed: {err}"
    assert user_data is not None
    assert user_data["email"] == test_email.lower()
    print(f"Registered user ID: {user_data['id']}, Email: {user_data['email']}")

    # Inspect directly in SQLite
    cursor.execute("SELECT id, email, password_hash, full_name, is_active FROM users WHERE id = ?", (user_data["id"],))
    row = cursor.fetchone()
    assert row is not None, "User record not found in SQLite data/agentic_ai.db!"
    db_id, db_email, db_hash, db_name, db_active = row
    assert db_email == test_email.lower()
    assert db_hash.startswith("$2b$12$"), "Database password_hash is not bcrypt!"
    assert db_name == "Nixitha Unta"
    print(f"[PASS] Verified SQLite record: Email='{db_email}', Hash='{db_hash[:20]}...', Name='{db_name}'")

    # 6. Duplicate Email Prevention
    print("\n--- 6. Testing Duplicate Registration Prevention ---")
    dup_ok, dup_user, dup_err = await auth_service.register_user(
        email=test_email,
        password=strong_pwd,
        confirm_password=strong_pwd
    )
    assert dup_ok is False, "Duplicate email registration was not prevented!"
    assert "already exists" in dup_err.lower()
    print(f"[PASS] Successfully prevented duplicate registration: {dup_err}")

    # 7. Authentication against Database
    print("\n--- 7. Testing User Authentication ---")
    auth_ok, auth_data, auth_err = await auth_service.authenticate_user(test_email, strong_pwd)
    assert auth_ok is True, f"Authentication failed with valid credentials: {auth_err}"
    assert auth_data["id"] == user_data["id"]
    print(f"[PASS] Authenticated successfully via email: {auth_data['email']}")

    auth_wrong_ok, auth_wrong_data, auth_wrong_err = await auth_service.authenticate_user(test_email, "WrongPass123!")
    assert auth_wrong_ok is False, "Authentication succeeded with wrong password!"
    print(f"[PASS] Successfully rejected wrong password: {auth_wrong_err}")

    conn.close()

    # 8. API Endpoint Testing via FastAPI TestClient
    print("\n--- 8. Testing FastAPI Auth Endpoints (Signup, Login, Me, Logout) ---")
    client = TestClient(test_app)

    # Test Signup API
    api_email = f"api_user_{uuid.uuid4().hex[:6]}@example.com"
    signup_res = client.post("/api/v1/auth/signup", json={
        "email": api_email,
        "password": "Compl3x@Password!",
        "confirm_password": "Compl3x@Password!",
        "full_name": "API Tester",
        "username": f"apitest_{uuid.uuid4().hex[:4]}"
    })
    assert signup_res.status_code == 200, f"Signup API failed: {signup_res.text}"
    signup_data = signup_res.json()
    assert signup_data["success"] is True
    assert signup_data["user"]["email"] == api_email
    assert "token" in signup_data
    created_id = signup_data["user"]["id"]
    print(f"✓ POST /api/v1/auth/signup returned 200 OK with session token.")

    # Test Signup with Weak Password
    weak_res = client.post("/api/v1/auth/signup", json={
        "email": "another@example.com",
        "password": "weak",
        "confirm_password": "weak"
    })
    assert weak_res.status_code == 400, f"Expected 400 for weak password, got: {weak_res.status_code}"
    print(f"✓ POST /api/v1/auth/signup correctly returned 400 Bad Request for weak password: {weak_res.json()['detail']}")

    # Test Login API
    login_res = client.post("/api/v1/auth/login", json={
        "email_or_username": api_email,
        "password": "Compl3x@Password!"
    })
    assert login_res.status_code == 200, f"Login API failed: {login_res.text}"
    login_data = login_res.json()
    assert login_data["success"] is True
    assert login_data["user"]["id"] == created_id
    print(f"✓ POST /api/v1/auth/login returned 200 OK.")

    # Test Login API with Wrong Password
    bad_login_res = client.post("/api/v1/auth/login", json={
        "email_or_username": api_email,
        "password": "WrongPassword99!"
    })
    assert bad_login_res.status_code == 401, f"Expected 401 for bad login, got: {bad_login_res.status_code}"
    print(f"✓ POST /api/v1/auth/login correctly returned 401 Unauthorized for invalid password.")

    # Test Me API
    me_res = client.get(f"/api/v1/auth/me?user_id={created_id}")
    assert me_res.status_code == 200, f"Me API failed: {me_res.text}"
    assert me_res.json()["email"] == api_email
    print(f"✓ GET /api/v1/auth/me returned 200 OK.")

    # Test Logout API
    logout_res = client.post("/api/v1/auth/logout")
    assert logout_res.status_code == 200, f"Logout API failed: {logout_res.text}"
    print(f"✓ POST /api/v1/auth/logout returned 200 OK.")

    print("\n========================================")
    print("ALL 8 VERIFICATION SUITES PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
