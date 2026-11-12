with open('frontend/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add Auth Guard in head
auth_guard = '''    <!-- Authentication Guard -->
    <script>
        (function enforceAuth() {
            const rawUser = localStorage.getItem('agentic_ai_user');
            if (!rawUser) {
                window.location.replace('/login');
                return;
            }
            try {
                const user = JSON.parse(rawUser);
                if (!user || !user.id) {
                    window.location.replace('/login');
                }
            } catch (e) {
                window.location.replace('/login');
            }
        })();
    </script>
</head>'''
content = content.replace('</head>', auth_guard, 1)

# 2. Update userAuthWidget in header
old_auth_widget_start = '<!-- User Auth Widget -->'
old_auth_widget_end = '</div>\n        </div>\n    </header>'
new_auth_widget = '''<!-- User Profile & Action Bar -->
            <div class="user-auth-widget" id="userAuthWidget" style="display:flex; align-items:center; gap:0.75rem;">
                <a href="/home" class="btn-secondary" style="padding:0.3rem 0.65rem; font-size:0.75rem; text-decoration:none; display:flex; align-items:center; gap:0.35rem;" title="View Project Overview & Features">
                    <span>🏠</span>
                    <span>Home</span>
                </a>
                <div id="authLoggedIn" style="display:flex; align-items:center; gap:0.6rem;">
                    <div class="user-profile-badge" style="display:flex; align-items:center; gap:0.5rem; background:var(--bg-card); padding:0.3rem 0.75rem; border-radius:20px; border:1px solid var(--border-color);">
                        <div class="user-avatar" id="userAvatar" style="width:24px; height:24px; border-radius:50%; background:linear-gradient(135deg, var(--primary), var(--accent-cyan)); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.75rem; color:#fff;">U</div>
                        <div style="display:flex; flex-direction:column; line-height:1.15; text-align:left;">
                            <span id="userDisplayName" style="font-size:0.8rem; font-weight:600; color:var(--text-main);">User</span>
                            <span id="userDisplayEmail" style="font-size:0.68rem; color:var(--text-muted);">user@example.com</span>
                        </div>
                    </div>
                    <button class="btn-secondary" id="btnLogout" onclick="logoutUser()" title="Sign Out" style="padding:0.3rem 0.65rem; font-size:0.75rem; cursor:pointer;">
                        🚪 Sign Out
                    </button>
                </div>
            </div>
        </div>
    </header>'''

idx1 = content.find(old_auth_widget_start)
idx2 = content.find(old_auth_widget_end)
if idx1 != -1 and idx2 != -1:
    content = content[:idx1] + new_auth_widget + content[idx2 + len(old_auth_widget_end):]
    print('Header updated successfully.')
else:
    print('Warning: header auth widget pattern not found.')

# 3. Remove embedded authModal
modal_start = '<!-- Authentication Modal (Sign In & Sign Up with Live Validation) -->'
modal_end = '<script src="/static/js/app.js"></script>'
m_idx1 = content.find(modal_start)
m_idx2 = content.find(modal_end)
if m_idx1 != -1 and m_idx2 != -1:
    content = content[:m_idx1] + '    ' + content[m_idx2:]
    print('Auth modal removed successfully.')
else:
    print('Warning: auth modal pattern not found.')

with open('frontend/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('frontend/dashboard.html generated successfully.')
