import os
import sys
import sqlite3

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'agentic_ai.db')

def print_sep(char='=', length=85):
    print(char * length)

def get_conn():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)

def overview():
    conn = get_conn()
    c = conn.cursor()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    
    print_sep('=')
    print("  AGENTIC AI PLATFORM - SQLITE DATABASE OVERVIEW")
    print(f"  Database File: {DB_PATH}")
    print_sep('=')
    print(f"  {'TABLE NAME':<25} | {'TOTAL ROWS':<12} | DESCRIPTION")
    print_sep('-')
    
    descriptions = {
        'users': 'Registered user accounts & profile metadata',
        'user_memories': 'Preferences (summary_style, signature, theme), facts & rules',
        'tasks': 'Autonomous workflow tasks executed by the 7-agent graph',
        'task_steps': 'Atomic step execution records, inputs, outputs, and status',
        'scheduled_jobs': 'Configured recurring APScheduler cron jobs',
        'conversation_messages': 'Multi-turn chat session history between user and AI',
        'workflow_logs': 'Granular real-time agent execution events and timestamps'
    }
    
    for t in tables:
        count = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        desc = descriptions.get(t, 'Custom database table')
        print(f"  {t:<25} | {count:<12} | {desc}")
    print_sep('=')
    print("\nCommands to inspect records:")
    print("  python view_database.py memories    -> View stored preferences & profile facts")
    print("  python view_database.py tasks       -> View recent executed workflows & statuses")
    print("  python view_database.py steps       -> View individual agent execution steps")
    print("  python view_database.py messages    -> View chat conversation message history")
    print("  python view_database.py jobs        -> View scheduled cron jobs")
    print("  python view_database.py users       -> View registered users")
    print("  python view_database.py query \"SQL\" -> Run custom SELECT query\n")
    conn.close()

def view_memories(limit=15):
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute(f"SELECT id, category, key, value, updated_at FROM user_memories ORDER BY updated_at DESC LIMIT {limit}"))
    print_sep('=')
    print(f"  STORED USER MEMORIES & PREFERENCES (Latest {len(rows)})")
    print_sep('=')
    for r in rows:
        print(f"  [{r[1].upper()}] Key: {r[2]}")
        print(f"     Value:      {r[3]}")
        print(f"     Updated At: {r[4]}")
        print_sep('-', 65)
    conn.close()

def view_tasks(limit=10):
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute(f"SELECT id, prompt, status, created_at FROM tasks ORDER BY created_at DESC LIMIT {limit}"))
    print_sep('=')
    print(f"  RECENT WORKFLOW TASKS (Latest {len(rows)})")
    print_sep('=')
    for r in rows:
        print(f"  Task ID:    {r[0]}")
        print(f"  Status:     {r[2]}")
        print(f"  Created At: {r[3]}")
        print(f"  Prompt:     {r[1]}")
        print_sep('-', 65)
    conn.close()

def view_steps(limit=10):
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute(f"SELECT step_index, agent_name, tool_name, status, description, created_at FROM task_steps ORDER BY created_at DESC LIMIT {limit}"))
    print_sep('=')
    print(f"  RECENT AGENT EXECUTION STEPS (Latest {len(rows)})")
    print_sep('=')
    for r in rows:
        print(f"  Step {r[0]} | Agent: {r[1]} | Tool: {r[2]} | Status: {r[3]}")
        print(f"  Description: {r[4]}")
        print(f"  Timestamp:   {r[5]}")
        print_sep('-', 65)
    conn.close()

def view_messages(limit=10):
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute(f"SELECT session_id, role, content, timestamp FROM conversation_messages ORDER BY timestamp DESC LIMIT {limit}"))
    print_sep('=')
    print(f"  RECENT CHAT CONVERSATION MESSAGES (Latest {len(rows)})")
    print_sep('=')
    for r in reversed(rows):
        content_preview = r[2][:160] + ('...' if len(r[2]) > 160 else '')
        print(f"  [{r[3]}] ({r[1].upper()}) [Session: {r[0]}]:")
        print(f"     {content_preview}")
        print_sep('-', 65)
    conn.close()

def view_jobs():
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute("SELECT id, name, cron_expression, prompt, is_active FROM scheduled_jobs ORDER BY created_at DESC"))
    print_sep('=')
    print(f"  SCHEDULED CRON WORKFLOWS ({len(rows)} jobs)")
    print_sep('=')
    for r in rows:
        status = "ACTIVE" if r[4] else "PAUSED"
        print(f"  Job: {r[1]} | Status: {status} | Cron: {r[2]}")
        print(f"  Prompt: {r[3]}")
        print_sep('-', 65)
    conn.close()

def view_users():
    conn = get_conn()
    c = conn.cursor()
    rows = list(c.execute("SELECT id, username, email, created_at FROM users"))
    print_sep('=')
    print(f"  REGISTERED USERS ({len(rows)} users)")
    print_sep('=')
    for r in rows:
        print(f"  ID: {r[0]} | Username: {r[1]} | Email: {r[2] or 'N/A'} | Created: {r[3]}")
    print_sep('=')
    conn.close()

def run_query(sql):
    conn = get_conn()
    c = conn.cursor()
    try:
        rows = list(c.execute(sql))
        cols = [d[0] for d in c.description] if c.description else []
        print_sep('=')
        print(f"  QUERY RESULT: {sql}")
        print_sep('=')
        print(' | '.join(cols))
        print_sep('-')
        for r in rows:
            print(' | '.join(str(val) for val in r))
        print(f"\nTotal rows returned: {len(rows)}")
    except Exception as e:
        print(f"SQL Query Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        overview()
    else:
        cmd = args[0].lower()
        if cmd in ['memories', 'memory', 'prefs', 'preferences']:
            view_memories()
        elif cmd in ['tasks', 'task', 'history']:
            view_tasks()
        elif cmd in ['steps', 'step']:
            view_steps()
        elif cmd in ['messages', 'chat', 'chats']:
            view_messages()
        elif cmd in ['jobs', 'schedules', 'sched']:
            view_jobs()
        elif cmd in ['users', 'user']:
            view_users()
        elif cmd == 'query' and len(args) > 1:
            run_query(' '.join(args[1:]))
        else:
            print(f"Unknown command: '{cmd}'")
            overview()
