# main.py
import os
import re
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from course_advisor_chatbot import ask_course_bot

load_dotenv(dotenv_path="./.env", override=True)

app = FastAPI(title="SkillCapital Chatbot", version="1.0.0")

# ------------------------- CORS -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- Request/Response Models -------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ------------------------- Markdown to HTML Conversion -------------------------
def markdown_to_html(md_text: str) -> str:
    """
    Converts Markdown links [text](url) to clickable HTML links <a href="url" target="_blank">text</a>
    """
    html_text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank">\1</a>',
        md_text
    )
    return html_text

# ------------------------- Serve HTML Frontend -------------------------
@app.get("/", response_class=HTMLResponse)
def serve_chatbot():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SkillCapital Support</title>
    <style>
      :root{--brand:#ff0f57; --brand-dark:#d10d47; --bg:#f5f7fb;
      --bot:#ffffff; --user:#ff0f57; --text:#20242b; --muted:#6b7280;
      --shadow:0 18px 40px rgba(0,0,0,0.12);}
      *{ box-sizing:border-box; }
      body{ margin:0; min-height:100vh; background:var(--bg); font-family: Inter, system-ui, Segoe UI, Roboto, sans-serif; display:flex; align-items:center; justify-content:center; padding:22px; }
      .widget{ width:100%; max-width:380px; height:620px; background:#fff; border-radius:18px; box-shadow:var(--shadow); display:flex; flex-direction:column; overflow:hidden; position:relative; }
      .header{ background:var(--brand); color:#fff; padding:14px 16px; display:flex; align-items:center; gap:10px; }
      .logo{ width:34px; height:34px; border-radius:50%; background:#fff1f5; color:var(--brand); display:grid; place-items:center; font-weight:800; }
      .title{ font-weight:700; }
      .status{ margin-left:auto; font-size:12px; display:flex; align-items:center; gap:8px; opacity:.95; }
      .dot{ width:8px; height:8px; border-radius:999px; background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,0.2); }
      .messages{ flex:1; padding:14px; background:#fafbff; overflow:auto; }
      .bubble{ max-width:78%; padding:10px 12px; border-radius:14px; font-size:14px; line-height:1.45; box-shadow:0 1px 0 rgba(0,0,0,0.04); }
      .bot{ background:var(--bot); color:var(--text); border:1px solid #eef0f6; }
      .user{ background:var(--user); color:#fff; margin-left:auto; }
      .row{ display:flex; gap:10px; margin:10px 0; align-items:flex-end; }
      .avatar{ width:28px; height:28px; border-radius:50%; background:#ffe4ec; color:var(--brand); display:grid; place-items:center; font-weight:800; }
      .composer{ padding:10px; border-top:1px solid #eef0f6; background:#fff; display:flex; gap:10px; align-items:center; }
      .input{ flex:1; display:flex; background:#f3f5fb; border-radius:999px; padding:8px 12px; border:1px solid #e8ebf4; }
      .input input{ flex:1; border:none; outline:none; background:transparent; font-size:14px; padding:6px 8px; }
      .send{ width:44px; height:44px; background:var(--brand); border:none; border-radius:50%; display:grid; place-items:center; color:#fff; cursor:pointer; box-shadow:0 8px 16px rgba(255,15,87,0.35); }
      .send:hover{ background:var(--brand-dark); }
      .hint{ position:absolute; bottom:70px; right:16px; font-size:11px; color:var(--muted); }
    </style>
  </head>
  <body>
    <div class="widget">
      <div class="header">
        <div class="logo">SC</div>
        <div class="title">SKILL CAPITAL SUPPORT</div>
        <div class="status"><span class="dot"></span>Online</div>
      </div>

      <div id="messages" class="messages"></div>

      <div class="composer">
        <div class="input"><input id="msg" placeholder="Type a message..." autocomplete="off"/></div>
        <button id="sendBtn" class="send" aria-label="Send" onclick="send()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M2 21l21-9L2 3v7l15 2-15 2v7z" fill="currentColor"/>
          </svg>
        </button>
      </div>
      <div class="hint">Press Enter to send</div>
    </div>

    <script>
      const messagesEl = document.getElementById('messages');
      const inputEl = document.getElementById('msg');
      const sendBtn = document.getElementById('sendBtn');

      function addBubble(text, role='bot'){
        const row = document.createElement('div');
        row.className = 'row';

        if(role==='bot'){
          const av = document.createElement('div');
          av.className = 'avatar';
          av.textContent = 'SC';
          row.appendChild(av);
        }

        const b = document.createElement('div');
        b.className = 'bubble ' + (role==='user' ? 'user' : 'bot');

        if(role==='user'){
          b.textContent = text;
        } else {
          b.innerHTML = text;  // render clickable links
        }

        row.appendChild(b);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      addBubble('👋 Hi User, welcome to Skill Capital! How can I assist you today?','bot');

      async function send(){
        const message = inputEl.value.trim();
        if(!message) return;
        inputEl.value='';
        addBubble(message,'user');
        sendBtn.disabled = true;
        try{
          const res = await fetch('/chat', { 
            method:'POST', 
            headers:{'Content-Type':'application/json'}, 
            body: JSON.stringify({ message }) 
          });
          if(!res.ok){ addBubble('Sorry, an error occurred.','bot'); return; }
          const data = await res.json();
          addBubble(data.reply || '');
        }catch(e){ addBubble('Sorry, I had trouble responding.','bot'); }
        finally{ sendBtn.disabled = false; inputEl.focus(); }
      }

      inputEl.addEventListener('keydown', (e)=>{
        if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
      });
    </script>
  </body>
</html>
"""

# ------------------------- Chat Endpoint -------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        reply, session_id = ask_course_bot(req.message, req.session_id)
        # Convert Markdown links to clickable HTML
        reply = markdown_to_html(reply)
        return ChatResponse(reply=reply, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
