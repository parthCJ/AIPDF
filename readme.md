# 💬 Chat with PDFs using DeepSeek (LangChain + Streamlit)

This Streamlit app allows users to upload a PDF and interactively **ask questions** about its content. It uses **LangChain**, **Ollama's DeepSeek model**, and **FAISS** to retrieve and answer queries based on the document.

---

## 🚀 Features

- Upload and analyze any PDF document.
- Chat interface to ask questions about the PDF.
- Uses `deepseek-r1:7b` model for LLM responses.
- Embeds PDF text using `OllamaEmbeddings` and stores in a FAISS vector store.
- Clean and intuitive UI with Streamlit.

---

## 🧠 Tech Stack

- **[Streamlit](https://streamlit.io/)** – Frontend UI
- **[LangChain](https://www.langchain.com/)** – Text splitting, document loading, prompt chaining
- **[Ollama](https://ollama.com/)** – Embedding & LLM (`deepseek-r1:7b`)
- **[FAISS](https://github.com/facebookresearch/faiss)** – Vector database for fast retrieval

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pdf-chat-deepseek.git
   cd pdf-chat-deepseek

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Make sure you have Ollama and the model::**
   ```bash
   ollama run deepseek-r1:7b

-----


## 📦 Running the app

2. **Make sure you're in the project directory, then run:**
   ```bash
   streamlit run app.py

-----

##  ## 🛠️ Built With

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [DeepSeek LLM](https://huggingface.co/deepseek-ai)
- [ChatGPT](https://openai.com/index/chatgpt/)

------
