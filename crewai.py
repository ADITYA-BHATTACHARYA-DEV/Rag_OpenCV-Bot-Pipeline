import os
import time
import base64
import cv2
import ollama
import numpy as np
import pytesseract
import re
from PIL import Image, ImageGrab
from typing import List, Dict, Any, Optional
from pathlib import Path
import pyautogui

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Simulated multi-agent structure
class VisionAgent:
    def __init__(self, model: str):
        self.model = model

    def analyze(self, image_path: str, query: str = "") -> str:
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            prompt = query or "Describe this image in detail."
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[encoded],
                options={"temperature": 0.2}
            )
            return response['response']
        except Exception as e:
            return f"[VisionAgent Error] {e}"

class OCRAgent:
    def extract_text(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            return f"[OCRAgent Error] {e}"

class RetrieverAgent:
    def __init__(self, text_data_dir: str, device: str):
        self.embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2",
            model_kwargs={"device": device}
        )
        self.vectorstore = FAISS.from_texts(["Initial doc"], self.embedder)
        self.load_knowledge(text_data_dir)

    def load_knowledge(self, directory: str):
        docs = []
        for file in Path(directory).rglob("*.txt"):
            try:
                loader = TextLoader(str(file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"[Text Load Error] {file}: {e}")
        for file in Path(directory).rglob("*.pdf"):
            try:
                loader = UnstructuredPDFLoader(str(file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"[PDF Load Error] {file}: {e}")
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            new_vs = FAISS.from_documents(splits, self.embedder)
            self.vectorstore.merge_from(new_vs)

    def retrieve(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=5)
            relevant = [doc.page_content for doc, score in docs if score < 0.5]
            return "\n\n".join(relevant)
        except Exception as e:
            return f"[RetrieverAgent Error] {e}"

class ReasoningAgent:
    def __init__(self, model: str):
        self.model = model

    def react_reason(self, query: str, image_info: str, ocr_info: str, rag_info: str) -> str:
        prompt = f"""
You are an expert assistant for eSim, an electronics simulation tool for PCB and circuit design.

Follow ReAct (Reason + Act) strategy:
1. Observe the user's query.
2. Use visual, OCR, and document context to form your reasoning.
3. Provide a final answer.

User Query:
{query}

Visual Analysis:
{image_info or 'N/A'}

OCR Extracted Text:
{ocr_info or 'N/A'}

Relevant Knowledge:
{rag_info or 'N/A'}

Answer in a concise and technically helpful manner.
"""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_ctx": 2048,
                    "top_k": 20,
                    "top_p": 0.9
                }
            )
            return response["response"]
        except Exception as e:
            return f"[ReasoningAgent Error] {e}"

class RewardAgent:
    def score_response(self, user_query: str, response: str) -> float:
        # Placeholder for RL reward mechanism
        # Can later be replaced with human-in-the-loop or rule-based scoring
        if "error" in response.lower():
            return -1.0
        elif "N/A" in response:
            return 0.0
        else:
            return 1.0  # Good response

# Main coordinator (Multi-agent ESIM Copilot)
class ESIMCopilot:
    def __init__(self, config: Dict[str, Any]):
        self.text_model = config.get("text_model", "qwen:1.8b")
        self.vision_model = config.get("vision_model", "qwen:1.8b")
        self.text_data_dir = config.get("text_data_dir", "./knowledge_base")
        self.device = config.get("device", "cpu")
        self.screenshot_dir = config.get("screenshot_dir", "./screenshots")

        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._ensure_ollama_running()
        self._pull_models()

        self.vision_agent = VisionAgent(self.vision_model)
        self.ocr_agent = OCRAgent()
        self.retriever_agent = RetrieverAgent(self.text_data_dir, self.device)
        self.reasoning_agent = ReasoningAgent(self.text_model)
        self.reward_agent = RewardAgent()

    def _ensure_ollama_running(self):
        try:
            ollama.list()
        except:
            print("Starting Ollama server...")
            os.system("ollama serve &")
            time.sleep(5)

    def _pull_models(self):
        for model in [self.text_model, self.vision_model]:
            try:
                ollama.show(model)
            except:
                print(f"Pulling model: {model}")
                ollama.pull(model)

    def _take_screenshot(self) -> Optional[str]:
        try:
            screen_width, screen_height = pyautogui.size()
            screenshot = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
            screenshot.save(path)
            return path
        except Exception as e:
            print(f"[Screenshot error] {e}")
            return None

    def process_query(self, query: str, image_path: Optional[str] = None) -> str:
        image_info, ocr_info = "", ""
        screen_keywords = ["screen", "diagram", "current view", "what's on the screen", "what you see"]

        if any(k in query.lower() for k in screen_keywords) and image_path is None:
            image_path = self._take_screenshot()
            if image_path:
                query = f"What is shown in this diagram? {query}"

        if image_path and os.path.exists(image_path):
            image_info = self.vision_agent.analyze(image_path, query)
            if any(k in query.lower() for k in ["text", "read", "label", "extract"]):
                ocr_info = self.ocr_agent.extract_text(image_path)

        rag_info = self.retriever_agent.retrieve(query)
        response = self.reasoning_agent.react_reason(query, image_info, ocr_info, rag_info)

        reward = self.reward_agent.score_response(query, response)
        print(f"[Reward Signal]: {reward}")
        return response

# Entry point
if __name__ == "__main__":
    config = {
        "text_model": "qwen:1.8b",
        "vision_model": "qwen:1.8b",
        "text_data_dir": "./knowledge_base",
        "device": "cpu",
        "screenshot_dir": "./screenshots"
    }

    copilot = ESIMCopilot(config)
    print("🧠 eSim Agentic Copilot is ready! Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.strip().lower() == "quit":
            break

        image_path = None
        if "image:" in query:
            query, image_path = query.split("image:")
            query = query.strip()
            image_path = image_path.strip()

        reply = copilot.process_query(query, image_path)
        print("\n🤖 Assistant:", reply)
