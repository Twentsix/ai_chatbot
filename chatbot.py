import os
import json
import requests
import numpy as np
import time
import sys
import logging
import uuid

from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_classic.chains.conversation.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
# --- Konfigurasi AI & Database ---
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIMENSION = EMBEDDING_MODEL.get_sentence_embedding_dimension()

OUT_OF_CONTEXT_MSG = "Maaf, informasi tersebut tidak tersedia dalam pusat data kami. Silakan tanyakan hal lain terkait layanan kami."
SYSTEM_PROMPT_BASE = (
    """
    Anda adalah customer service yang ramah, artikulatif, dan profesional yang mewakili perusahaan di situs webnya.
    Anda berinteraksi langsung dengan pengunjung yang sebagian besar baru mengenal perusahaan 
    dan belum terbiasa dengan layanan yang ditawarkan.
    Tugas Anda adalah memberikan informasi yang akurat, bermanfaat, 
    dan terpercaya yang bersumber secara eksklusif dari dokumentasi perusahaan dan Knowledge Base (RAG).
    
    Misi Utama:
    1. Memandu pengunjung untuk memahami perusahaan, produk, layanan, 
    dan nilai unik (unique value propositions) hanya dengan merujuk pada 
    informasi yang diambil dari Knowledge Base (RAG) atau database perusahaan.
    2. Jika tidak ada di database, katakan bahwa itu di luar konteks. Dan jangan sebut 'database'.
    3. Mendukung berbagai pertanyaan — mulai dari pertanyaan dasar 
    hingga kebutuhan skenario yang kompleks—yang berakar secara eksklusif 
    pada dokumentasi perusahaan yang terverifikasi.
    4. Mohon untuk tidak menjawab pertanyaan sensitif mengenai terorisme dan pornografi.
    
    Protokol Perilaku:
    1. Kepatuhan Ketat pada Knowledge Base atau Database Perusahaan
        - Jawab hanya dengan konten yang diambil melalui RAG dari Knowledge Base atau database perusahaan.
        - Jangan pernah membuat jawaban berdasarkan asumsi, pengetahuan eksternal, atau pelatihan sebelumnya.
        - Jangan pernah memanipulasi, menebak, atau mengekstrapolasi di luar apa yang terkandung secara eksplisit dalam dokumentasi.
        - Jika informasi relevan tidak ditemukan, jawab dengan jelas:
            "Pertanyaan yang bagus — Saya tidak dapat menemukan informasi tersebut dalam dokumentasi perusahaan."
        - Jika pengguna mencari informasi di luar Knowledge Base, 
        tunjukkan ketidakhadiran informasi tersebut dengan sopan dan tawarkan bantuan lainnya.
    
    2. Keahlian Perusahaan yang Komprehensif
        - Siap menangani berbagai topik relevan bisnis, termasuk: 
        Ringkasan & misi perusahaan, produk/layanan, segmen pelanggan, 
        panduan penggunaan, budaya tim, hingga artikel bantuan.
        - Jika informasi terperinci tidak tersedia, gunakan pesan fallback yang jelas 
        dan tawarkan bantuan untuk topik lain.
    
    4. Penanganan Skenario Kompleks & Kasus Khusus
        - Untuk pertanyaan multi-langkah atau berbasis skenario, 
        berikan panduan langkah-demi-langkah secara ketat sesuai garis besar dokumentasi.
        - Jika pertanyaan mencakup beberapa topik, strukturkan respons secara jelas 
        dan referensikan setiap area yang relevan.
    
    5. Keamanan, Privasi, dan Kepercayaan
        - Jangan pernah meminta atau memproses kata sandi, informasi pembayaran, atau data pribadi pengguna.
    
    6. Batas Penjelasan (Fallback Strategis)
        - Jika Anda sudah tidak bisa menjelaskan lebih lanjut atau 
        informasi memang tidak tersedia sama sekali di sistem, sampaikan:
            "Mohon maaf, terkait hal tersebut silakan hubungi nomor kontak kami dan alamat email kami"
    
    7. Pengalaman Pengguna yang Dioptimalkan
        - Libatkan pengguna dalam percakapan yang alami, merespons secara kontekstual, 
        dan antisipasi kebutuhan informasi mereka berikutnya 
        berdasarkan alur perjalanan pelanggan yang terdokumentasi.
    
    Ringkasan:
    Anda adalah perwakilan digital yang terpercaya—memberdayakan pengunjung 
    dan membina interaksi yang bermakna melalui panduan berbasis dokumentasi yang presisi. 
    Anda mengoptimalkan setiap interaksi untuk akurasi, kejelasan, dan nilai tambah.
    """)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
DB_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeBaseManager:
    def __init__(self, db_url: str, embedding_model: SentenceTransformer, vector_dim: int):
        self.engine = create_engine(db_url)
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
    
    def initialize_database(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE TABLE IF NOT EXISTS knowledge ());"))
                conn.execute(text("CREATE TABLE IF NOT EXISTS history ();"))
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"DB Init Error: {e}")
            return False

    def insert_knowledge(self, content: str):
        embedding = self.embedding_model.encode([content])[0]
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        with self.engine.connect() as conn:
            conn.execute(
                text("INSERT INTO knowledge"),
                {"c": content, "e": embedding_str})
            conn.commit()
    
    def get_all_knowledge(self):
        """Mengambil semua isi database untuk ditampilkan di dashboard"""
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT FROM knowledge")).fetchall()
                return [{"id": row[0], "content": row[1]} for row in res]
        except Exception as e:
            return []
    
    def retrieve_context(self, query: str):
        print(datetime.now(), "fungsi retrieve context")
        embedding = self.embedding_model.encode([query])[0]
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        try:
            with self.engine.connect() as conn:
                # Menggunakan cosine distance <=> dan limit 3
                res = conn.execute(
                    text("SELECT FROM knowledge"), 
                    {"e": embedding_str, "q": query}
                ).fetchall()
                
                #return " ".join([row[0] for row in res])
                contexts = [row[0] for row in res]
                
        except Exception as e:
            logging.error(f"Retrieve Error: {e}")
            return ""

    def save_chat(self, session_id, role, message):
        with self.engine.connect() as conn:
            conn.execute(
                text("INSERT INTO history"),{}
            )
            conn.commit()

    def get_history(self, session_id, limit=1):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT FROM history"),
                {"u": session_id, "l": limit}
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(result)]

kb_manager = None

def get_memory(session_id):
    """Mengambil memori LangChain yang terhubung ke Postgres"""
    print(datetime.now(), "fungsi get memory")
    history = PostgresChatMessageHistory(connection_string,session_id,table_name)
    
    # 2. Bungkus ke dalam objek Memory LangChain
    memory = ConversationBufferMemory(chat_memory,return_messages,memory_key)
    #return ConversationBufferMemory(chat_memory=history, return_messages=True)
    return memory

def generate_ai_response(system_prompt, history):
    print(datetime.now(), "fungsi generate ai response")
    messages = [{"role": "system", "content": system_prompt}] + history
    try:
        r = requests.post(
            f"{OLLAMA_API_BASE}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "messages": messages,
                "stream": False
                ,"options": # Konfigurasi Parameter AI
                    {"num_ctx","num_thread","temperature","top_p","num_predict","stop"}
            },
            timeout=200
        )
        return r.json().get("message", {}).get("content", "Sistem sibuk.")
    except Exception as e:
        logging.error(f"Ollama Error: {e}")
        return "Gagal menghubungi AI."

def get_chat_history_with_memory(session_id):
    print(datetime.now(), "fungsi get chat history with memory")
    """
    Mengambil memori percakapan dari PostgreSQL menggunakan LangChain.
    Setiap session_id akan memiliki riwayat chat terpisah di tabel 'message_store'.
    """
    history = PostgresChatMessageHistory(connection_string,session_id,table_name)
    
    # Menggunakan ConversationBufferMemory untuk mengelola riwayat
    memory = ConversationBufferMemory(chat_memory,return_messages,memory_key)
    return memory

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot Widget</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .knowledge-card:hover { border-color: #3b82f6; }
        .chat-window { height: 400px; overflow-y: auto; scroll-behavior: smooth; }
        .message-bubble { max-width: 80%; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px; }
        .bot { background: #f1f5f9; align-self: flex-start; border-bottom-left-radius: 2px; }
        .user { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
    </style>
</head>

<body class="bg-gray-100 font-sans">
    <!-- Header / Navbar -->
    <nav class="bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center sticky top-0 z-10">
        <div class="flex items-center gap-3">
            <div class="bg-blue-600 p-2 rounded-lg text-white">
                <i class="fa-solid fa-database"></i>
            </div>
            <h1 class="text-xl font-bold tracking-tight">AI Chatbot <span class="text-blue-600">Knowledge Hub</span></h1>
        </div>
        <div class="flex items-center gap-4 text-sm font-medium text-slate-500">
            <span class="flex items-center gap-1"><i class="fa-solid fa-circle text-green-500 text-[10px]"></i> AI Online</span>
            <span class="border-l pl-4">v2.0 Beta</span>
        </div>
    </nav>
    
    <main class="flex-1 p-6 max-w-7xl mx-auto w-full">
        <div class="flex flex-col gap-6">
            <!-- Header Konten -->
            <div class="flex justify-between items-end">
                <div>
                    <h2 class="text-2xl font-bold">Dokumentasi Terdaftar</h2>
                    <p class="text-slate-500">Isi Knowledge Base yang saat ini tersimpan di database RAG.</p>
                </div>
                <button onclick="location.reload()" class="bg-white border px-4 py-2 rounded-lg text-sm hover:bg-slate-50 transition">
                    <i class="fa-solid fa-rotate mr-2"></i>Refresh Data
                </button>
            </div>

            <!-- Grid Konten dari Database -->
            <div id="knowledgeGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Data akan dimuat di sini oleh JS -->
                <div class="animate-pulse bg-slate-200 h-32 rounded-xl"></div>
                <div class="animate-pulse bg-slate-200 h-32 rounded-xl"></div>
                <div class="animate-pulse bg-slate-200 h-32 rounded-xl"></div>
            </div>
        </div>
    </main>
    
    <!-- Chatbot Widget -->
    <div class="fixed bottom-5 right-5 flex flex-col items-end">
        <div id="chatContainer" class="hidden w-80 md:w-96 bg-white rounded-2xl shadow-2xl flex flex-col mb-4 overflow-hidden border border-gray-200">
            <div class="bg-blue-600 p-4 text-white font-bold flex justify-between items-center">
                <span>AI Customer Support</span>
                <button onclick="toggleChat()" class="text-white hover:text-gray-200">✕</button>
            </div>
            <div id="chatWindow" class="chat-window p-4 flex flex-col bg-gray-50">
                <div class="message-bubble bot text-sm">
                Halo! Selamat datang. Ada yang bisa saya bantu hari ini?
                </div>
            </div>
            <div class="p-3 border-t bg-white flex items-center gap-2">
                <input type="text" id="userInput" placeholder="Ketik pesan..." class="flex-1 border rounded-full px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                <button onclick="sendMessage()" class="bg-blue-600 text-white p-2 rounded-full hover:bg-blue-700">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" /></svg>
                </button>
            </div>
        </div>
        <button onclick="toggleChat()" class="bg-blue-600 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition-all scale-110">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
        </button>
    </div>

    <script>
        // Load Data Dashboard
        async function loadDashboard() {
            const grid = document.getElementById('knowledgeGrid');
            try {
                const res = await fetch('/api/knowledge');
                const data = await res.json();
                grid.innerHTML = '';
                
                if(data.length === 0) {
                    grid.innerHTML = '<div class="col-span-full text-center py-20 text-slate-400 italic">Database kosong. Silakan jalankan python.py</div>';
                    return;
                }

                data.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'knowledge-card bg-white border border-slate-200 p-5 rounded-xl shadow-sm transition-all hover:shadow-md';
                    card.innerHTML = `
                        <div class="flex justify-between items-start mb-3">
                            <span class="text-[10px] font-bold bg-slate-100 text-slate-500 px-2 py-1 rounded">ID: ${item.id}</span>
                            <i class="fa-solid fa-file-lines text-slate-300"></i>
                        </div>
                        <p class="text-sm text-slate-600 leading-relaxed">${item.content}</p>
                    `;
                    grid.appendChild(card);
                });
            } catch (e) {
                grid.innerHTML = '<p class="text-red-500">Gagal memuat data.</p>';
            }
        }
        
        // Membuat Session ID jika belum ada
        let sessionId = localStorage.getItem('chat_session_id');
        if (!sessionId) {
            sessionId = 'sess-' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('chat_session_id', sessionId);
        }
        
        const chatWindow = document.getElementById('chatWindow');
        const userInput = document.getElementById('userInput');

        function toggleChat() {
            document.getElementById('chatContainer').classList.toggle('hidden');
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if(!text) return;
            appendMessage(text, 'user');
            userInput.value = '';

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text, session_id: sessionId, user_id: 'user'})
                });
                const data = await res.json();
                appendMessage(data.response, 'bot');
            } catch {
                appendMessage('Maaf, gangguan koneksi.', 'bot');
            }
        }

        function appendMessage(text, sender) {
            const div = document.createElement('div');
            div.className = `message-bubble ${sender} text-sm`;
            div.innerText = text;
            chatWindow.appendChild(div);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        userInput.addEventListener('keypress', (e) => { if(e.key === 'Enter') sendMessage(); });
        
        // Initial load
        loadDashboard();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/knowledge")
def get_knowledge():
    """Endpoint untuk mengambil semua data knowledge base"""
    return jsonify(kb_manager.get_all_knowledge())

@app.route("/chat", methods=["POST"])
def chat():
    print(datetime.now(), "fungsi chat")
    data = request.json
    session_id = data.get("session_id", "default_sess")
    user_text = data.get("message", "").strip()

    if not session_id:
        return jsonify({"response": "Error: Session ID missing"}), 400

    if not user_text:
        return jsonify({"response": "Pesan kosong."})
    
    # 1. Ambil Memori (Riwayat Chat dari Postgres)
    memory = get_memory(session_id)
    chat_history_messages = memory.load_memory_variables({})["history"]
    
    # Format history untuk Ollama
    formatted_history = []
    for msg in chat_history_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted_history.append({"role": role, "content": msg.content})

    # 2. Ambil Konteks RAG (Pengetahuan dari Database)
    context = kb_manager.retrieve_context(user_text)

    # 3. Rakit Prompt
    full_system_prompt = f"{SYSTEM_PROMPT_BASE}\n\nKONTEKS PENGETAHUAN:\n{context}"
    print(datetime.now(), "setelah full_system_prompt")
    # 4. Panggil AI Ollama
    try:
        print(datetime.now(), "gabung semua messages sebelum kirim ke ollama")
        messages = [
            {"role": "system", "content": full_system_prompt}
        ] + formatted_history + [
            {"role": "user", "content": user_text}
        ]
        
        print(datetime.now(), "lempar ke ollama")
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/chat",
            json={"model": DEFAULT_MODEL, "messages": messages, "stream": False},
            timeout=2000
        )
        ai_response = response.json().get("message", {}).get("content", "Maaf, saya kesulitan memproses itu.")

        # 5. SIMPAN ke Memori
        memory.save_context({"input": user_text}, {"output": ai_response})

    except Exception as e:
        logging.error(f"Error: {e}")
        ai_response = "Gagal menghubungi AI."

    print(datetime.now(), "akhir fungsi chat")
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    kb_manager = KnowledgeBaseManager(DB_URL, EMBEDDING_MODEL, VECTOR_DIMENSION)
    kb_manager.initialize_database()
    app.run(host,port)