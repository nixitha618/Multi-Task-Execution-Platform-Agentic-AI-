import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app

@pytest.mark.asyncio
async def test_api_tasks():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create Task
        res = await client.post("/api/v1/tasks", json={
            "prompt": "Calculate 15 * 8 - 20",
            "user_id": "test_user_api"
        })
        assert res.status_code == 200
        data = res.json()
        assert "id" in data
        task_id = data["id"]
        assert data["prompt"] == "Calculate 15 * 8 - 20"

        # List Tasks
        list_res = await client.get("/api/v1/tasks")
        assert list_res.status_code == 200
        tasks = list_res.json()
        assert len(tasks) > 0

        # Get Single Task
        task_res = await client.get(f"/api/v1/tasks/{task_id}")
        assert task_res.status_code == 200
        assert task_res.json()["id"] == task_id

@pytest.mark.asyncio
async def test_api_rag_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/documents/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"

@pytest.mark.asyncio
async def test_api_memory():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create Memory
        res = await client.post("/api/v1/memory", json={
            "user_id": "test_user_api",
            "category": "preference",
            "key": "report_format",
            "value": "Always format tables cleanly with markdown",
            "importance": 0.9
        })
        assert res.status_code == 200
        mem = res.json()
        assert mem["key"] == "report_format"
        mem_id = mem["id"]

        # List Memories
        list_res = await client.get("/api/v1/memory?user_id=test_user_api")
        assert list_res.status_code == 200
        mems = list_res.json()
        assert any(m["id"] == mem_id for m in mems)

        # Delete Memory
        del_res = await client.delete(f"/api/v1/memory/{mem_id}")
        assert del_res.status_code == 200

@pytest.mark.asyncio
async def test_api_schedule():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/schedule", json={
            "user_id": "test_user_api",
            "name": "Daily Sales Audit",
            "prompt": "Analyze sales and generate summary report",
            "cron_expression": "0 8 * * *"
        })
        assert res.status_code == 200
        job = res.json()
        assert job["name"] == "Daily Sales Audit"

        list_res = await client.get("/api/v1/schedule")
        assert list_res.status_code == 200
        assert len(list_res.json()) > 0

@pytest.mark.asyncio
async def test_dashboard_served():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "Agentic AI" in res.text

        dash_res = await client.get("/dashboard")
        assert dash_res.status_code == 200
        assert "Autonomous Multi-Task Execution Platform" in dash_res.text

        css_res = await client.get("/static/css/styles.css")
        assert css_res.status_code == 200

        js_res = await client.get("/static/js/app.js")
        assert js_res.status_code == 200
