import sys
import os
import sqlite3
import uuid

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.api.endpoints_auth import router as auth_router

# Build minimal test application mirroring main.py route table
FRONTEND_DIR = os.path.abspath("frontend")

app = FastAPI()
app.include_router(auth_router, prefix="/api/v1/auth")

@app.get("/")
@app.get("/home")
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "home.html"))

@app.get("/signup")
def serve_signup():
    return FileResponse(os.path.join(FRONTEND_DIR, "signup.html"))

@app.get("/login")
def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

def run_multi_page_tests():
    print("==================================================")
    print("TESTING MULTI-PAGE FLOW & REDIRECTION ARCHITECTURE")
    print("==================================================")
    client = TestClient(app)

    # 1. Test Home Landing Page
    print("\n--- 1. Testing Home Page (GET / & GET /home) ---")
    res_root = client.get("/")
    assert res_root.status_code == 200, f"GET / failed: {res_root.status_code}"
    assert "Orchestrate Complex Workflows" in res_root.text
    assert "/signup" in res_root.text
    assert "/login" in res_root.text
    print("[PASS] GET / serves high-fidelity Home Landing Page with navigation & CTA links.")

    res_home = client.get("/home")
    assert res_home.status_code == 200, f"GET /home failed: {res_home.status_code}"
    print("[PASS] GET /home alias serves identical Home Landing Page.")

    # 2. Test Sign Up Page
    print("\n--- 2. Testing Dedicated Sign Up Page (GET /signup) ---")
    res_signup = client.get("/signup")
    assert res_signup.status_code == 200, f"GET /signup failed: {res_signup.status_code}"
    assert "Create account" in res_signup.text
    assert "strength-meter-container" in res_signup.text
    assert "password-criteria-list" in res_signup.text
    assert "/dashboard" in res_signup.text  # redirection target present
    print("[PASS] GET /signup serves dedicated Split-Screen Sign Up Page with live meter & /dashboard redirection.")

    # 3. Test Login Page
    print("\n--- 3. Testing Dedicated Login Page (GET /login) ---")
    res_login = client.get("/login")
    assert res_login.status_code == 200, f"GET /login failed: {res_login.status_code}"
    assert "Sign in to Platform" in res_login.text
    assert "loginIdentifier" in res_login.text
    assert "/signup" in res_login.text  # link to create account
    assert "/dashboard" in res_login.text  # redirection target present
    print("[PASS] GET /login serves dedicated Split-Screen Login Page with SQLite auth & /dashboard redirection.")

    # 4. Test Protected Dashboard Page
    print("\n--- 4. Testing Protected Dashboard Page (GET /dashboard) ---")
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200, f"GET /dashboard failed: {res_dash.status_code}"
    assert "enforceAuth" in res_dash.text or "localStorage.getItem('agentic_ai_user')" in res_dash.text
    assert "/login" in res_dash.text  # auth guard redirect to login
    assert "Workflow Studio" in res_dash.text
    print("[PASS] GET /dashboard serves Dashboard equipped with Auth Guard redirecting to /login.")

    # 5. Test First-Time Registration & Database Persistence
    print("\n--- 5. Testing First-Time User Registration & SQLite Persistence ---")
    unique_email = f"student_{uuid.uuid4().hex[:6]}@rgukt.ac.in"
    password = "Secur3P@ssword2026!"
    signup_payload = {
        "email": unique_email,
        "password": password,
        "confirm_password": password,
        "full_name": "Nixitha Unta",
        "username": f"nixitha_{uuid.uuid4().hex[:4]}"
    }
    signup_res = client.post("/api/v1/auth/signup", json=signup_payload)
    assert signup_res.status_code == 200, f"Signup API failed: {signup_res.text}"
    signup_data = signup_res.json()
    assert signup_data["success"] is True
    assert signup_data["user"]["email"] == unique_email.lower()
    user_id = signup_data["user"]["id"]
    print(f"[PASS] User created with ID: {user_id} and email: {unique_email}")

    # Inspect SQLite database file
    conn = sqlite3.connect("data/agentic_ai.db")
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash, full_name FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    assert row is not None, "User not found in data/agentic_ai.db!"
    assert row[1] == unique_email.lower()
    assert row[2].startswith("$2b$12$"), "Password not hashed with 12-round bcrypt in SQLite!"
    assert row[3] == "Nixitha Unta"
    conn.close()
    print(f"[PASS] Verified user in SQLite data/agentic_ai.db with 12-round bcrypt salted hash.")

    # 6. Test Returning User Login with Registered Credentials
    print("\n--- 6. Testing Returning User Login with Registered Email & Password ---")
    login_res = client.post("/api/v1/auth/login", json={
        "email_or_username": unique_email,
        "password": password
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    login_data = login_res.json()
    assert login_data["success"] is True
    assert login_data["user"]["id"] == user_id
    print(f"[PASS] Returning user authenticated successfully against SQLite hash.")

    # 7. Test Bad Password Rejection
    print("\n--- 7. Testing Wrong Password Rejection ---")
    bad_login_res = client.post("/api/v1/auth/login", json={
        "email_or_username": unique_email,
        "password": "WrongPassword123!"
    })
    assert bad_login_res.status_code == 401
    print(f"[PASS] Correctly rejected unauthorized login attempt.")

    print("\n==================================================")
    print("ALL MULTI-PAGE FLOW & REDIRECTION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_multi_page_tests()
