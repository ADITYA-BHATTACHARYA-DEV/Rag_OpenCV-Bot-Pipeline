import os
import time
import base64
import cv2
import ollama
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from pathlib import Path
import pytesseract
import pyautogui
from concurrent.futures import ThreadPoolExecutor

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
class MultimodalAssistant:
    def __init__(self, config: Dict[str, Any]):
        self.text_model = config.get("text_model", "phi:2.7b-chat-v2-q4_0")
        self.vision_model = config.get("vision_model", "smollm2")  # Use quantized version
        self.text_data_dir = config.get("text_data_dir", "./knowledge_base")
        self.device = config.get("device", "cpu")
        self.quantized = config.get("quantized", True)

        self._ensure_ollama_running()
        self._pull_models()
        self._init_text_embedder()

        self.image_prompts = {
            'default': "Describe this image concisely.",
            'what_is': "What is in this image? Describe the main objects, people, and actions.",
            'text_in_image': "Extract and summarize any visible text in this image."
        }

    def _ensure_ollama_running(self):
        try:
            ollama.list(timeout=2)
        except:
            print("Starting optimized Ollama server...")
            os.system("nice -n 10 ollama serve --num-parallel 1 &")
            time.sleep(3)

    def _pull_models(self):
        models_to_pull = []

        for model in [self.text_model, self.vision_model]:
            try:
                ollama.show(model)
            except:
                pull_model = model + (":q4_0" if self.quantized else "")
                models_to_pull.append(pull_model)

        if models_to_pull:
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.map(ollama.pull, models_to_pull)

    def _init_text_embedder(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )
        self.vectorstore = None
        if os.path.exists(self.text_data_dir):
            self._load_knowledge_base()

    def _load_knowledge_base(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", ".", " "]
        )
        docs = []
        files = list(Path(self.text_data_dir).rglob("*.[tT][xX][tT]")) + \
                list(Path(self.text_data_dir).rglob("*.[pP][dD][fF]"))

        for file in files:
            try:
                if file.suffix.lower() == ".txt":
                    loader = TextLoader(str(file))
                else:
                    loader = UnstructuredPDFLoader(str(file), mode="elements", strategy="fast")
                docs.extend(loader.load())
            except Exception as e:
                print(f"[Load Error] {file}: {e}")

        if docs:
            splits = splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(
                splits,
                self.embedder,
                distance_strategy="dot"
            )

    def _optimize_image(self, image_path: str) -> str:
        img = Image.open(image_path)
        if max(img.size) > 512:
            img.thumbnail((512, 512))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        temp_path = "/tmp/opt_img.jpg"
        img.save(temp_path, "JPEG", quality=85)
        return temp_path

    def _analyze_image(self, image_path: str, query: str = "") -> str:
        try:
            query_lower = query.lower()
            if 'what is' in query_lower or 'what\'s in' in query_lower:
                prompt = self.image_prompts['what_is']
            elif 'text' in query_lower or 'read' in query_lower:
                prompt = self.image_prompts['text_in_image']
            else:
                prompt = self.image_prompts['default']

            temp_path = self._optimize_image(image_path)
            with open(temp_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            response = ollama.generate(
                model=self.vision_model,
                prompt=prompt,
                images=[encoded],
                options={"temperature": 0.3, "num_ctx": 1024, "num_predict": 150}
            )
            return response['response']
        except Exception as e:
            return f"[Image analysis failed] {str(e)[:100]}"

    def _ocr_image(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""
            if max(img.shape) > 1024:
                scale = 1024 / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--oem 1 --psm 6')

            return text.strip() if text.strip() else self._analyze_image(image_path, "Extract all text from this image.")
        except Exception as e:
            return self._analyze_image(image_path, "Extract all text from this image.")

    def _process_text_only(self, query: str) -> str:
        try:
            response = ollama.generate(
                model=self.text_model,
                prompt=query,
                options={"temperature": 0.3, "num_predict": 60, "top_k": 30, "top_p": 0.8}
            )
            return response['response']
        except Exception as e:
            return f"[Error] {str(e)[:100]}"

    def _retrieve_rag_info(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=1)
            return "\n".join([doc.page_content for doc in docs])
        except:
            return ""

    def process_query(self, query: str, image_path: Optional[str] = None) -> str:
        # Check if image is referenced but not provided
        image_keywords = ['screen', 'image', 'picture', 'screenshot', 'see here', 'look at', 'what is shown', 'what is wrong']
        if not image_path and any(kw in query.lower() for kw in image_keywords):
            try:
                print("[!] No image provided. Capturing screenshot automatically...")
                screenshot = pyautogui.screenshot()
                image_path = "/tmp/screenshot_query.jpg"
                screenshot.save(image_path)
            except Exception as e:
                return f"[Error capturing screenshot] {str(e)}"

        if not image_path and len(query.split()) < 5:
            return self._process_text_only(query)

        image_info, ocr_info = "", ""
        rag_info = self._retrieve_rag_info(query) if self.vectorstore else ""

        if image_path and os.path.exists(image_path):
            needs_ocr = any(word in query.lower() for word in ['text', 'read', 'label', 'word', 'letter'])

            with ThreadPoolExecutor(max_workers=2) as executor:
                image_future = executor.submit(self._analyze_image, image_path, query)
                ocr_future = executor.submit(self._ocr_image, image_path) if needs_ocr else None

                image_info = image_future.result()
                ocr_info = ocr_future.result() if ocr_future else ""

        context_parts = []
        if image_info:
            context_parts.append(f"Visual Context: {image_info}")
        if ocr_info:
            context_parts.append(f"Extracted Text: {ocr_info}")
        if rag_info:
            context_parts.append(f"Reference Information: {rag_info}")

        context = "\n".join(context_parts) if context_parts else "No additional context"
        final_prompt = f"User question: {query}\n\n{context}\n\nProvide a concise answer:"

        try:
            response = ollama.generate(
                model=self.text_model,
                prompt=final_prompt,
                options={"temperature": 0.4, "num_ctx": 1024, "num_predict": 120, "top_k": 20, "top_p": 0.9}
            )
            return response['response']
        except Exception as e:
            return f"[Error generating response] {str(e)[:100]}"


# --- CLI Interface ---
if __name__ == "__main__":
    config = {
        "text_model": "llama2:7b",
        "vision_model": "llava:7b",
        "text_data_dir": "./knowledge_base",
        "device": "cpu",
        "quantized": True
    }

    assistant = MultimodalAssistant(config)
    print("Multimodal Assistant ready. Type 'quit' to exit.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'quit':
            break

        image_path = None
        if "image:" in query:
            parts = query.split("image:")
            query = parts[0].strip()
            image_path = parts[1].strip() if len(parts) > 1 else None

        response = assistant.process_query(query, image_path)
        print(f"\nAssistant: {response}")
