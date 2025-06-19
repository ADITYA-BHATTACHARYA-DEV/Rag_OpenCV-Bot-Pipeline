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



from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter





class ESIMCopilot:

    def __init__(self, config: Dict[str, Any]):

        self.text_model = config.get("text_model", "qwen:1.8b")

        self.vision_model = config.get("vision_model", "qwen:1.8b")

        self.text_data_dir = config.get("text_data_dir", "./knowledge_base")

        self.device = config.get("device", "cpu")

        self.screenshot_dir = config.get("screenshot_dir", "./screenshots")

        

        # Create screenshot directory if it doesn't exist

        os.makedirs(self.screenshot_dir, exist_ok=True)



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

            model_name="intfloat/e5-small-v2",

            model_kwargs={"device": self.device}

        )

        self.vectorstore = FAISS.from_texts(["Initial doc"], self.embedder)



        if os.path.exists(self.text_data_dir):

            docs = []



            for file in Path(self.text_data_dir).rglob("*.txt"):

                try:

                    loader = TextLoader(str(file))

                    docs.extend(loader.load())

                except Exception as e:

                    print(f"[Text Load Error] {file}: {e}")



            for file in Path(self.text_data_dir).rglob("*.pdf"):

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

            docs = self.vectorstore.similarity_search_with_score(query, k=5)

            relevant = [doc.page_content for doc, score in docs if score < 0.5]

            return "\n\n".join(relevant)

        except Exception as e:

            return f"[RAG error] {e}"



    def _extract_components(self, text: str) -> List[str]:

        return re.findall(r'(R\\d+\\s*=\\s*[\\d.]+[kKmMuU]?)', text)



    def _take_screenshot(self) -> Optional[str]:

        """Take a screenshot of the current screen and save it to the screenshot directory."""

        try:

            # Get screen dimensions

            screen_width, screen_height = pyautogui.size()

            

            # Take screenshot

            screenshot = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))

            

            # Generate filename with timestamp

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")

            

            # Save screenshot

            screenshot.save(screenshot_path)

            return screenshot_path

        except Exception as e:

            print(f"[Screenshot error] {e}")

            return None



    def process_query(self, query: str, image_path: Optional[str] = None) -> str:

        image_info, ocr_info = "", ""

        

        # Check if user is asking about the screen/diagram and no image is provided

        screen_keywords = ["screen", "diagram", "current view", "what's on the screen", "what you see"]

        if (any(keyword in query.lower() for keyword in screen_keywords) and 

            image_path is None):

            image_path = self._take_screenshot()

            if image_path:

                query = f"What is shown in this diagram? {query}"



        if image_path and os.path.exists(image_path):

            image_info = self._analyze_image(image_path, query)

            if any(k in query.lower() for k in ["text", "read", "label", "extract"]):

                ocr_info = self._ocr_image(image_path)



        rag_info = self._retrieve_rag_info(query)



        prompt = f"""

You are an expert assistant for eSim, an electronics simulation tool used for PCB and circuit design.



Only use the provided documents and extracted information. Do not make up any details.



User Question:

{query}



Visual Analysis:

{image_info or "N/A"}



OCR Text:

{ocr_info or "N/A"}



Knowledge Base Context:

{rag_info or "No relevant context found."}



Answer in a precise, technical, and helpful manner.

"""



        try:

            response = ollama.generate(

                model=self.text_model,

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

            return f"[Response generation error] {e}"





if __name__ == "__main__":

    config = {

        "text_model": "qwen:1.8b",

        "vision_model": "qwen:1.8b",

        "text_data_dir": "./knowledge_base",

        "device": "cpu",

        "screenshot_dir": "./screenshots"

    }



    copilot = ESIMCopilot(config)

    print("eSim Agentic Copilot ready. Type 'quit' to exit.")



    while True:

        query = input("\nYou: ")

        if query.strip().lower() == "quit":

            break



        image_path = None

        if "image:" in query:

            query, image_path = query.split("image:")

            image_path = image_path.strip()

            query = query.strip()



        response = copilot.process_query(query, image_path)

        print("\nAssistant:", response)