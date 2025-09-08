import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from course_advisor_chatbot import course_advi_rag
# Your RAG + memory chatbot   

app = FastAPI(title="SkillCapital Chatbot", version="1.0.0")

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# Serve frontend directly from HTML+CSS string
@app.get("/", response_class=HTMLResponse)
def serve_chatbot():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SkillCapital Course Advisor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;}
.chat-container{width:90%;max-width:800px;height:600px;background:white;border-radius:20px;box-shadow:0 20px 40px rgba(0,0,0,0.1);display:flex;flex-direction:column;overflow:hidden;}
.chat-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;text-align:center;}
.chat-header h1{font-size:24px;margin-bottom:5px;}
.chat-header p{opacity:0.9;font-size:14px;}
.chat-messages{flex:1;padding:20px;overflow-y:auto;background:#f8f9fa;}
.message{margin-bottom:15px;display:flex;align-items:flex-start;}
.message.user{justify-content:flex-end;}
.message.bot{justify-content:flex-start;}
.message-content{max-width:70%;padding:12px 16px;border-radius:18px;word-wrap:break-word;}
.message.user .message-content{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;}
.message.bot .message-content{background:white;border:1px solid #e0e0e0;color:#333;}
.chat-input{padding:20px;background:white;border-top:1px solid #e0e0e0;display:flex;gap:10px;}
.chat-input input{flex:1;padding:12px 16px;border:1px solid #ddd;border-radius:25px;outline:none;font-size:14px;}
.chat-input input:focus{border-color:#667eea;}
.chat-input button{padding:12px 24px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;border-radius:25px;cursor:pointer;font-size:14px;font-weight:600;transition:transform 0.2s;}
.chat-input button:hover{transform:translateY(-2px);}
.chat-input button:disabled{opacity:0.6;cursor:not-allowed;transform:none;}
.typing-indicator{display:none;padding:12px 16px;background:white;border:1px solid #e0e0e0;border-radius:18px;color:#666;font-style:italic;}
.welcome-message{text-align:center;color:#666;font-style:italic;margin-top:50px;}
</style>
</head>
<body>
<div class="chat-container">
<div class="chat-header">
<h1>🎓 SkillCapital Course Advisor</h1>
<p>Your AI-powered course guidance assistant</p>
</div>
<div class="chat-messages" id="chatMessages">
<div class="welcome-message">👋 Hi! I'm your SkillCapital course advisor. Ask me about courses, pricing, or enrollment!</div>
</div>
<div class="chat-input">
<input type="text" id="messageInput" placeholder="Ask about SkillCapital courses..." autocomplete="off">
<button id="sendButton" onclick="sendMessage()">Send</button>
</div>
</div>
<script>
function addMessage(content,isUser=false){const m=document.getElementById('chatMessages');const d=document.createElement('div');d.className=`message ${isUser?'user':'bot'}`;const c=document.createElement('div');c.className='message-content';c.textContent=content;d.appendChild(c);m.appendChild(d);m.scrollTop=m.scrollHeight;}
function showTyping(){const m=document.getElementById('chatMessages');const d=document.createElement('div');d.className='message bot';d.id='typingIndicator';const c=document.createElement('div');c.className='typing-indicator';c.textContent='course_advisor is typing...';c.style.display='block';d.appendChild(c);m.appendChild(d);m.scrollTop=m.scrollHeight;}
function hideTyping(){const t=document.getElementById('typingIndicator');if(t)t.remove();}
async function sendMessage(){const input=document.getElementById('messageInput');const button=document.getElementById('sendButton');const message=input.value.trim();if(!message)return;addMessage(message,true);input.value='';button.disabled=true;showTyping();try{const response=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:message})});if(!response.ok)throw new Error(`HTTP error! status: ${response.status}`);const data=await response.json();hideTyping();addMessage(data.reply);}catch(e){hideTyping();addMessage('Sorry, I encountered an error. Please try again.');console.error(e);}finally{button.disabled=false;input.focus();}}
document.getElementById('messageInput').addEventListener('keypress',function(e){if(e.key==='Enter')sendMessage();});document.getElementById('messageInput').focus();
</script>
</body>
</html>
"""

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SkillCapital Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        reply = course_advi_rag(req.message.strip())  # RAG + memory + prompt
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
