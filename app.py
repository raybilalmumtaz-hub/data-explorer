import os
import gradio as gr
import pandas as pd
import numpy as np
import requests
import pickle
import time

# ========== CONFIGURATION ==========
# Get API key from Hugging Face Secrets (or environment variable)
GROQ_API_KEY = os.environ.get("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", "")

# Your knowledge base documents (public Google Drive links)
GOOGLE_DRIVE_LINKS = [
    "https://drive.google.com/file/d/1GI0dqwyN-T9J4zU4Wz6DChvacGYuYUIO/view?usp=sharing"
]

# ========== GLOBAL VARIABLES ==========
kb_chunks = []

# ========== SIMPLE EMBEDDING FUNCTION ==========
def simple_embedding(text):
    """Simple text embedding for demonstration"""
    # Convert text to lowercase and remove punctuation
    text = str(text).lower()
    words = text.split()
    
    # Create a simple frequency-based embedding (for demo)
    # In production, you'd use a proper embedding model
    embedding = np.zeros(50)  # 50-dimensional embedding
    
    for i, word in enumerate(words[:50]):  # Use first 50 words
        # Simple hash-based embedding
        hash_val = hash(word) % 1000 / 1000.0
        embedding[i % 50] += hash_val
    
    # Normalize
    if np.linalg.norm(embedding) > 0:
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

# ========== KNOWLEDGE BASE LOADING ==========
def load_knowledge_base():
    """Load knowledge base from Google Drive"""
    global kb_chunks
    
    print("🔄 Loading knowledge base...")
    all_chunks = []
    
    for link in GOOGLE_DRIVE_LINKS:
        try:
            # Download CSV from Google Drive
            file_id = link.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Download with retry
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Read CSV from response content
                        import io
                        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
                        break
                except:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            
            print(f"✅ Loaded {len(df)} rows from Google Drive")
            
            # Clean data
            df = df.dropna(axis=1, thresh=len(df)*0.5)
            df = df.fillna("")
            
            # Create chunks
            for _, row in df.iterrows():
                # Combine row data
                row_text = " | ".join([
                    f"{col}: {val}" 
                    for col, val in row.items() 
                    if str(val).strip() and str(val).lower() != "nan"
                ])
                
                if len(row_text) > 30:
                    # Split into smaller chunks if needed
                    words = row_text.split()
                    for i in range(0, len(words), 150):
                        chunk = " ".join(words[i:i+150])
                        if len(chunk) > 50:
                            all_chunks.append(chunk[:500])
                            
        except Exception as e:
            print(f"⚠️ Error loading {link}: {str(e)}")
            # Add fallback data
            all_chunks.append("Knowledge base contains data from uploaded documents.")
            all_chunks.append("Ask questions about the available information.")
    
    kb_chunks = all_chunks if all_chunks else ["No data available."]
    print(f"📚 Created {len(kb_chunks)} knowledge chunks")
    return len(kb_chunks)

# ========== SIMPLE SIMILARITY SEARCH ==========
def search_chunks(question, chunks, top_k=3):
    """Simple similarity search without FAISS"""
    if not chunks:
        return []
    
    # Get question embedding
    q_embedding = simple_embedding(question)
    
    # Calculate similarities
    similarities = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = simple_embedding(chunk)
        similarity = np.dot(q_embedding, chunk_embedding)
        similarities.append((similarity, i, chunk))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Return top matches
    return [chunk for _, _, chunk in similarities[:top_k]]

# ========== ANSWER GENERATION ==========
def get_answer(question):
    """Get answer from knowledge base"""
    if not question or not question.strip():
        return "Please enter a question."
    
    if not kb_chunks:
        return "Knowledge base is still loading. Please wait a moment."
    
    try:
        # Search for relevant chunks
        relevant_chunks = search_chunks(question, kb_chunks, top_k=3)
        
        if not relevant_chunks:
            return "❌ I do not know. This information is not in my knowledge base."
        
        # Combine context
        context = "\n---\n".join(relevant_chunks)
        
        # If Groq API is available, use it
        if GROQ_API_KEY:
            try:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                
                prompt = f"""Answer the question using ONLY the information below. 
If the answer is not in the information, say exactly: "I do not know."
INFORMATION:
{context}
QUESTION: {question}
ANSWER (be concise, use only information above):"""
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Check if answer says "don't know"
                if any(phrase in answer.lower() for phrase in ["i do not know", "i don't know", "not in the"]):
                    return "❌ I do not know."
                
                return f"✅ {answer}"
                
            except ImportError:
                pass  # Groq not installed
            except Exception as e:
                print(f"⚠️ Groq error: {e}")
        
        # Fallback: Return most relevant chunk
        answer_text = relevant_chunks[0]
        if len(answer_text) > 300:
            answer_text = answer_text[:300] + "..."
        
        return f"📄 From knowledge base:\n\n{answer_text}"
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return "❌ Error processing question. Please try again."

# ========== GRADIO INTERFACE ==========
def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(
        title="Knowledge Base Assistant",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 700px; margin: auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .answer-box { 
            border-radius: 10px; 
            padding: 20px; 
            background: #f0f8ff;
            border: 2px solid #4CAF50;
            margin-top: 20px;
        }
        .info-text { font-size: 14px; color: #666; }
        """
    ) as app:
        
        with gr.Column(elem_classes="container"):
            # Header
            gr.Markdown("""
            <div class="header">
            <h1>📚 Knowledge Base Assistant</h1>
            <p>Ask questions about your Google Drive documents</p>
            </div>
            """)
            
            # Status
            kb_status = f"✅ Knowledge Base: {len(kb_chunks)} chunks loaded"
            gr.Markdown(f"**{kb_status}**")
            
            if not GROQ_API_KEY:
                gr.Markdown("""
                <div class="info-text">
                ℹ️ <b>Note:</b> Running in basic mode. 
                Add <code>GROQ_API_KEY</code> in Settings → Secrets for LLM answers.
                </div>
                """)
            
            # Question Input
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know?",
                    lines=2,
                    scale=4
                )
            
            # Buttons
            with gr.Row():
                submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            # Answer Output
            answer_output = gr.Markdown(
                label="Answer",
                value="*Your answer will appear here...*",
                elem_classes="answer-box"
            )
            
            # Examples
            with gr.Accordion("💡 Try these questions:", open=False):
                gr.Examples(
                    examples=[
                        ["What information is available?"],
                        ["Summarize the data"],
                        ["What topics are covered?"]
                    ],
                    inputs=question_input,
                    label="Example questions"
                )
            
            # Footer
            gr.Markdown("""
            ---
            <div class="info-text">
            <small>
            📁 <b>Source:</b> Google Drive documents<br>
            🔍 <b>Search:</b> Semantic similarity matching<br>
            🔒 <b>Policy:</b> Answers only from provided documents
            </small>
            </div>
            """)
        
        # Event handlers
        submit_btn.click(
            fn=get_answer,
            inputs=question_input,
            outputs=answer_output
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Your answer will appear here...*"),
            outputs=[question_input, answer_output]
        )
        
        # Enter key support
        question_input.submit(
            fn=get_answer,
            inputs=question_input,
            outputs=answer_output
        )
    
    return app

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("🚀 Starting Knowledge Base Assistant...")
    
    # Load knowledge base
    try:
        num_chunks = load_knowledge_base()
        print(f"✅ Loaded {num_chunks} knowledge chunks")
    except Exception as e:
        print(f"⚠️ Error loading knowledge base: {e}")
        kb_chunks = ["Knowledge base could not be loaded. Please check your Google Drive links."]
    
    # Create and launch interface
    try:
        app = create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        
        # Simple fallback interface
        with gr.Blocks() as fallback_app:
            gr.Markdown("# ❌ Application Error")
            gr.Markdown(f"Error: {str(e)}")
            gr.Markdown("Please check the logs for details.")
        fallback_app.launch()
