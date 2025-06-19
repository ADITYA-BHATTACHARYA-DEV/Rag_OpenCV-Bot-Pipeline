import os
import time
import base64
import cv2
import ollama
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from pathlib import Path

import pytesseract
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MultimodalAssistant:
    def __init__(self, config: Dict[str, Any]):
        self.text_model = config.get("text_model", "qwen:1.8b")
        self.vision_model = config.get("vision_model", "qwen:1.8b")  # Use a valid model here
        self.text_data_dir = config.get("text_data_dir", "./knowledge_base")
        self.device = config.get("device", "cpu")

        self._ensure_ollama_running()
        self._pull_models()
        self._init_text_embedder()

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

    def _init_text_embedder(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )
        self.vectorstore = FAISS.from_texts(["Initial doc"], self.embedder)

        if os.path.exists(self.text_data_dir):
            docs = []

            # Load .txt files
            txt_loader = TextLoader
            txt_files = Path(self.text_data_dir).rglob("*.txt")
            for file in txt_files:
                try:
                    loader = txt_loader(str(file))
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[Text Load Error] {file}: {e}")

            # Load .pdf files
            pdf_files = Path(self.text_data_dir).rglob("*.pdf")
            for file in pdf_files:
                try:
                    loader = UnstructuredPDFLoader(str(file))
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[PDF Load Error] {file}: {e}")

            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = splitter.split_documents(docs)
                new_vs = FAISS.from_documents(splits, self.embedder)
                self.vectorstore.merge_from(new_vs)

    def _ocr_image(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            return f"[OCR error] {e}"

    def _analyze_image(self, image_path: str, query: Optional[str] = "") -> str:
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            prompt = query or "Describe this image in detail."
            response = ollama.generate(
                model=self.vision_model,
                prompt=prompt,
                images=[encoded],
                options={"temperature": 0.2}
            )
            return response['response']
        except Exception as e:
            return f"[Image analysis failed] {e}"

    def _retrieve_rag_info(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=2)
            return "\n".join([doc.page_content for doc in docs])
        except:
            return ""

    def process_query(self, query: str, image_path: Optional[str] = None) -> str:
        image_info, ocr_info = "", ""

        if image_path and os.path.exists(image_path):
            image_info = self._analyze_image(image_path, query)
            if any(k in query.lower() for k in ["text", "read", "label"]):
                ocr_info = self._ocr_image(image_path)

        rag_info = self._retrieve_rag_info(query)

        prompt_parts = [
            f"User question: {query}",
            f"\nVisual Analysis:\n{image_info}" if image_info else "",
            f"\nOCR Text:\n{ocr_info}" if ocr_info else "",
            f"\nRelevant Docs:\n{rag_info}" if rag_info else ""
        ]
        final_prompt = "\n".join([part for part in prompt_parts if part.strip()])

        try:
            response = ollama.generate(
                model=self.text_model,
                prompt=final_prompt,
                options={
                    "temperature": 0.5,
                    "num_ctx": 2048,
                    "top_k": 20,
                    "top_p": 0.9
                }
            )
            return response["response"]
        except Exception as e:
            return f"[Error during response generation] {e}"

# --- Main CLI Interface ---
if __name__ == "__main__":
    config = {
        "text_model": "qwen:1.8b",
        "vision_model": "qwen:1.8b",  # Use the same valid model here
        "text_data_dir": "./knowledge_base",
        "device": "cpu"
    }

    assistant = MultimodalAssistant(config)

    print("Multimodal Assistant ready. Type 'quit' to exit.")
    while True:
        query = input("\nYou: ")
        if query.strip().lower() == "quit":
            break

        image_path = None
        if "image:" in query:
            query, image_path = query.split("image:")
            image_path = image_path.strip()
            query = query.strip()

        response = assistant.process_query(query, image_path)
        print(f"\nAssistant: {response}")
