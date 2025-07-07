import os
import time
import base64
import asyncio
import cv2
import ollama
import numpy as np
import pytesseract
import pyautogui
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import json
from collections import deque

from PIL import Image, ImageGrab
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, Dict, Any, Coroutine


@dataclass
class AgentResponse:
    content: str
    confidence: float
    metadata: dict

class VisionAgent:
    def __init__(self, model: str, cache_size: int = 100):
        self.model = model
        self.cache = LRUCache(cache_size)
        
    async def analyze(self, image_path: str, query: str = "") -> AgentResponse:
        cache_key = f"vision_{hash_file(image_path)}_{query}"
        if cached := self.cache.get(cache_key):
            return cached
            
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            
            prompt = query or "Describe this image in detail, focusing on technical diagrams, schematics, and electronic components."
            
            response = await ollama.async_generate(
                model=self.model,
                prompt=prompt,
                images=[encoded],
                options={
                    "temperature": 0.1,
                    "top_k": 30,
                    "num_ctx": 4096
                }
            )
            
            result = AgentResponse(
                content=response['response'],
                confidence=0.9,  # Vision models typically high confidence
                metadata={
                    'model': self.model,
                    'image_hash': hash_file(image_path),
                    'processing_time': response.get('total_duration', 0)
                }
            )
            
            self.cache.put(cache_key, result)
            return result
            
        except Exception as e:
            return AgentResponse(
                content=f"[VisionAgent Error] {e}",
                confidence=0.0,
                metadata={'error': str(e)}
            )

class OCRAgent:
    def __init__(self, cache_size: int = 200):
        self.cache = LRUCache(cache_size)
        self.preprocess_pipeline = [
            self._convert_to_grayscale,
            self._apply_adaptive_threshold,
            self._remove_noise
        ]
    
    async def extract_text(self, image_path: str) -> AgentResponse:
        cache_key = f"ocr_{hash_file(image_path)}"
        if cached := self.cache.get(cache_key):
            return cached
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
                
            # Apply preprocessing pipeline
            for process in self.preprocess_pipeline:
                img = process(img)
                
            # Custom config for technical documents
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:()[]{}<>-=+*/\\&|^%$#@!~`\'"'
            text = pytesseract.image_to_string(img, config=custom_config)
            
            # Post-processing
            text = self._post_process_text(text)
            
            # Estimate confidence (simple heuristic)
            confidence = min(0.7 + (len(text.split()) * 0.01), 0.95)
            
            result = AgentResponse(
                content=text.strip(),
                confidence=confidence,
                metadata={
                    'preprocessing': [p.__name__ for p in self.preprocess_pipeline],
                    'tesseract_config': custom_config
                }
            )
            
            self.cache.put(cache_key, result)
            return result
            
        except Exception as e:
            return AgentResponse(
                content=f"[OCRAgent Error] {e}",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def _apply_adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    def _post_process_text(self, text: str) -> str:
        # Fix common OCR errors in technical documents
        replacements = {
            'voltagc': 'voltage',
            'curren1': 'current',
            'resislor': 'resistor',
            'capaci1or': 'capacitor'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
            
        return text

class RetrieverAgent:
    def __init__(self, text_data_dir: str, device: str, cache_size: int = 500):
        self.embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",  # Upgraded to base model
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = FAISS.from_texts(["Initial doc"], self.embedder)
        self.cache = LRUCache(cache_size)
        self.load_knowledge(text_data_dir)
    
    def load_knowledge(self, directory: str):
        docs = []
        
        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            txt_files = list(Path(directory).rglob("*.txt"))
            pdf_files = list(Path(directory).rglob("*.pdf"))
            
            # Process text files
            txt_results = list(executor.map(self._load_text_file, txt_files))
            # Process PDF files
            pdf_results = list(executor.map(self._load_pdf_file, pdf_files))
            
        docs.extend([doc for sublist in txt_results + pdf_results for doc in sublist])
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Slightly larger chunks for technical docs
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = splitter.split_documents(docs)
            
            # Use FAISS with HNSW for faster search
            new_vs = FAISS.from_documents(
                splits, 
                self.embedder,
                distance_strategy="COSINE",
                hnsw_config={"M": 32, "efConstruction": 200}
            )
            self.vectorstore.merge_from(new_vs)
    
    def _load_text_file(self, file_path: Path) -> List[Any]:
        try:
            loader = TextLoader(str(file_path), autodetect_encoding=True)
            return loader.load()
        except Exception as e:
            print(f"[Text Load Error] {file_path}: {e}")
            return []
    
    def _load_pdf_file(self, file_path: Path) -> List[Any]:
        try:
            loader = UnstructuredPDFLoader(
                str(file_path),
                mode="elements",
                strategy="fast"
            )
            return loader.load()
        except Exception as e:
            print(f"[PDF Load Error] {file_path}: {e}")
            return []
    
    async def retrieve(self, query: str) -> AgentResponse:
        cache_key = f"retrieve_{hash(query)}"
        if cached := self.cache.get(cache_key):
            return cached
            
        try:
            # First do a fast, approximate search
            docs = self.vectorstore.similarity_search_with_score(
                query, 
                k=10,  # Get more results initially
                filter=None,
                search_type="similarity",
                score_threshold=0.7
            )
            
            # Filter and rerank
            relevant = [
                (doc.page_content, score) 
                for doc, score in docs 
                if score < 0.6
            ]
            
            if not relevant:
                return AgentResponse(
                    content="No relevant documents found.",
                    confidence=0.3,
                    metadata={'num_docs': 0}
                )
                
            # Sort by score (lower is better)
            relevant.sort(key=lambda x: x[1])
            
            # Take top 3 most relevant
            content = "\n\n".join([doc for doc, _ in relevant[:3]])
            
            # Confidence based on best match score
            best_score = relevant[0][1]
            confidence = max(0, 1 - best_score)  # Convert distance to confidence
            
            result = AgentResponse(
                content=content,
                confidence=confidence,
                metadata={
                    'num_docs': len(relevant),
                    'best_score': best_score,
                    'query': query
                }
            )
            
            self.cache.put(cache_key, result)
            return result
            
        except Exception as e:
            return AgentResponse(
                content=f"[RetrieverAgent Error] {e}",
                confidence=0.0,
                metadata={'error': str(e)}
            )

class ReasoningAgent:
    def __init__(self, model: str, react_optimizer_path: str = None):
        self.model = model
        self.react_optimizer = ReactOptimizer.load(react_optimizer_path) if react_optimizer_path else None
        self.temperature = 0.3
        self.top_k = 30
        self.max_react_iterations = 5
    
    async def reason(self, query: str, vision: AgentResponse, ocr: AgentResponse, context: AgentResponse) -> AgentResponse:
        # Dynamic temperature based on confidence of inputs
        avg_confidence = (vision.confidence + ocr.confidence + context.confidence) / 3
        dynamic_temp = max(0.1, min(0.5, 0.5 - (avg_confidence * 0.4)))
        
        prompt = self._build_prompt(query, vision, ocr, context)
        
        if self.react_optimizer:
            return await self._reason_with_react(query, prompt, dynamic_temp)
        else:
            return await self._simple_reason(prompt, dynamic_temp)
    
    async def _simple_reason(self, prompt: str, temperature: float) -> AgentResponse:
        try:
            response = await ollama.async_generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_k": self.top_k,
                    "num_ctx": 4096
                }
            )
            
            return AgentResponse(
                content=response['response'],
                confidence=0.8,  # Base confidence for LLM responses
                metadata={
                    'model': self.model,
                    'temperature': temperature,
                    'strategy': 'direct'
                }
            )
        except Exception as e:
            return AgentResponse(
                content=f"[ReasoningAgent Error] {e}",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _reason_with_react(self, query: str, initial_prompt: str, temperature: float) -> AgentResponse:
        history = []
        current_prompt = initial_prompt
        confidence = 0.7  # Starting confidence
        
        for i in range(self.max_react_iterations):
            try:
                response = await ollama.async_generate(
                    model=self.model,
                    prompt=current_prompt,
                    options={
                        "temperature": max(0.1, temperature * (1 - (i * 0.1))),  # Cool down over iterations
                        "top_k": self.top_k,
                        "num_ctx": 4096,
                        "stop": ["</final_answer>"]
                    }
                )
                
                response_text = response['response']
                history.append(response_text)
                
                if "<final_answer>" in response_text:
                    # Extract final answer
                    final_answer = response_text.split("<final_answer>")[-1].strip()
                    confidence = min(0.95, confidence + 0.05 * i)  # Slightly increase confidence with each step
                    
                    return AgentResponse(
                        content=final_answer,
                        confidence=confidence,
                        metadata={
                            'model': self.model,
                            'temperature': temperature,
                            'strategy': 'react',
                            'iterations': i + 1,
                            'history': history
                        }
                    )
                
                # Get next action from optimizer
                next_action = self.react_optimizer.get_next_action(query, response_text)
                current_prompt = f"{current_prompt}\n\n{response_text}\n\n{next_action}"
                
            except Exception as e:
                return AgentResponse(
                    content=f"[ReAct Reasoning Error] {e}",
                    confidence=max(0, confidence - 0.1),
                    metadata={
                        'error': str(e),
                        'iterations': i,
                        'history': history
                    }
                )
        
        # If we get here, we hit max iterations without final answer
        return AgentResponse(
            content="[ReAct Warning] Maximum iterations reached without final answer. Last response:\n" + history[-1],
            confidence=0.5,
            metadata={
                'model': self.model,
                'warning': 'max_iterations',
                'history': history
            }
        )
    
    def _build_prompt(self, query: str, vision: AgentResponse, ocr: AgentResponse, context: AgentResponse) -> str:
        prompt_template = """You are an expert assistant for eSim (Electronic Simulation Tool). Your task is to help with circuit design, simulation, and analysis.

Query: {query}

{vision_section}

{ocr_section}

{context_section}

{reasoning_instructions}"""
        
        vision_section = (
            "🖼 Visual Analysis (Confidence: {:.1f}%):\n{}\n".format(vision.confidence * 100, vision.content)
            if vision.content and "N/A" not in vision.content and "Error" not in vision.content else
            "🖼 No visual analysis available."
        )
        
        ocr_section = (
            "📝 Extracted Text (Confidence: {:.1f}%):\n{}\n".format(ocr.confidence * 100, ocr.content)
            if ocr.content and "N/A" not in ocr.content and "Error" not in ocr.content else
            "📝 No text extracted from image."
        )
        
        context_section = (
            "📚 Relevant Documentation (Confidence: {:.1f}%):\n{}\n".format(context.confidence * 100, context.content)
            if context.content and "No relevant" not in context.content else
            "📚 No relevant documentation found."
        )
        
        reasoning_instructions = (
            "Reason step-by-step. Consider all available information. "
            "If uncertain, ask clarifying questions or state assumptions.\n"
            "For circuit questions, provide: component values, connections, and expected behavior.\n"
            "For simulation questions, suggest: parameters to check, possible issues, and solutions."
        )
        
        if self.react_optimizer:
            reasoning_instructions += (
                "\n\nUse ReAct framework:\n"
                "Thought: <your analysis>\n"
                "Action: <next step or information needed>\n"
                "Observation: <result of action>\n"
                "...\n"
                "<final_answer>Your complete answer</final_answer>"
            )
        
        return prompt_template.format(
            query=query,
            vision_section=vision_section,
            ocr_section=ocr_section,
            context_section=context_section,
            reasoning_instructions=reasoning_instructions
        )

class RewardAgent:
    def __init__(self, reward_model_path: str = None):
        self.reward_model_path = reward_model_path
        self.reward_cache = {}
        self.min_reward = -1.0
        self.max_reward = 1.0
    
    def score(self, query: str, response: AgentResponse) -> float:
        cache_key = f"reward_{hash(query)}_{hash(response.content)}"
        if cache_key in self.reward_cache:
            return self.reward_cache[cache_key]
        
        # Basic heuristics
        if "error" in response.content.lower():
            reward = self.min_reward
        elif "N/A" in response.content or "not available" in response.content.lower():
            reward = 0.0
        else:
            # More sophisticated scoring
            reward = self._calculate_reward(query, response)
        
        self.reward_cache[cache_key] = reward
        return reward
    
    def _calculate_reward(self, query: str, response: AgentResponse) -> float:
        """Calculate reward based on multiple factors"""
        base_reward = 0.5  # Neutral starting point
        
        # Length factor (neither too short nor too long)
        length_factor = min(1.0, len(response.content.split()) / 50)
        
        # Confidence factor from the agent
        confidence_factor = response.confidence
        
        # Specificity factor (how many technical terms match the query)
        query_terms = set(query.lower().split())
        response_terms = set(response.content.lower().split())
        specificity = len(query_terms & response_terms) / max(1, len(query_terms))
        
        # Structure factor (well-formatted responses score higher)
        structure_score = 0.0
        if "\n" in response.content:  # Has some structure
            structure_score = 0.2
        if any(c in response.content for c in [":", "- ", "*"]):  # List-like
            structure_score = 0.4
        
        # Combine factors
        reward = base_reward + (
            0.2 * length_factor +
            0.3 * confidence_factor +
            0.3 * specificity +
            0.2 * structure_score
        )
        
        return min(self.max_reward, max(self.min_reward, reward))

class ReactOptimizer:
    """Reinforcement Learning optimized ReAct handler"""
    def __init__(self, policy_path: str = None):
        self.policy = self._load_policy(policy_path) if policy_path else self._default_policy()
        self.action_history = deque(maxlen=100)
    
    @classmethod
    def load(cls, path: str) -> 'ReactOptimizer':
        try:
            with open(path, 'r') as f:
                policy = json.load(f)
            return cls(policy)
        except:
            return cls()
    
    def _default_policy(self) -> Dict[str, Any]:
        return {
            "default": {
                "action_sequence": [
                    "Analyze the query and identify key components",
                    "Check available information sources",
                    "Formulate hypothesis",
                    "Verify with available data",
                    "Synthesize final answer"
                ],
                "weights": [0.2, 0.2, 0.3, 0.2, 0.1]
            },
            "circuit_design": {
                "action_sequence": [
                    "Identify circuit components",
                    "Check component connections",
                    "Verify circuit laws apply",
                    "Simulate expected behavior",
                    "Propose improvements"
                ],
                "weights": [0.3, 0.3, 0.2, 0.1, 0.1]
            }
        }
    
    def get_next_action(self, query: str, current_state: str) -> str:
        """Get the next optimal action based on current state"""
        query_type = self._classify_query(query)
        policy = self.policy.get(query_type, self.policy["default"])
        
        # Simple strategy: rotate through actions based on policy
        if not hasattr(self, '_action_index'):
            self._action_index = 0
        
        action = policy["action_sequence"][self._action_index % len(policy["action_sequence"])]
        self._action_index += 1
        
        self.action_history.append((query, current_state, action))
        return f"Action: {action}\nObservation:"
    
    def _classify_query(self, query: str) -> str:
        """Simple query classifier"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["circuit", "schematic", "component"]):
            return "circuit_design"
        if any(word in query_lower for word in ["simulat", "parameter", "analysis"]):
            return "simulation"
        return "default"
    
    def update_policy(self, reward: float):
        """Update policy based on reward (simple implementation)"""
        # In a full implementation, this would update the policy weights
        pass

class LRUCache:
    """Simple LRU Cache implementation"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
    
    def get(self, key: str) -> Any:
        if key in self.cache:
            self.order.remove(key)
            self.order.appendleft(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop()
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.appendleft(key)

def hash_file(filepath: str) -> str:
    """Quick file hash for caching"""
    return str(os.path.getmtime(filepath)) + str(os.path.getsize(filepath))

class ESIMCopilot:
    def __init__(self, config: Dict[str, Any]):
        self.text_model = config.get("text_model", "qwen:1.8b")
        self.vision_model = config.get("vision_model", "qwen:1.8b")
        self.text_data_dir = config.get("text_data_dir", "./knowledge_base")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.screenshot_dir = config.get("screenshot_dir", "./screenshots")
        self.react_policy_path = config.get("react_policy_path")
        self.reward_model_path = config.get("reward_model_path")
        
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._start_ollama()
        self._pull_models()
        
        # Initialize agents with caching
        self.vision_agent = VisionAgent(self.vision_model)
        self.ocr_agent = OCRAgent()
        self.retriever_agent = RetrieverAgent(self.text_data_dir, self.device)
        self.reasoning_agent = ReasoningAgent(self.text_model, self.react_policy_path)
        self.reward_agent = RewardAgent(self.reward_model_path)
        
        # Performance monitoring
        self.query_times = deque(maxlen=100)
        self.rewards = deque(maxlen=100)
        
        # Load performance optimizers
        self._load_performance_optimizers()
    
    def _start_ollama(self):
        try:
            ollama.list()
        except:
            print("Starting Ollama...")
            os.system("ollama serve &")
            time.sleep(5)
    
    def _pull_models(self):
        models_to_pull = set([self.text_model, self.vision_model])
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for model in models_to_pull:
                try:
                    ollama.show(model)
                except:
                    print(f"Pulling model {model}...")
                    futures.append(executor.submit(ollama.pull, model))
            
            for future in futures:
                future.result()  # Wait for all pulls to complete
    
    def _load_performance_optimizers(self):
        """Load any performance optimization modules"""
        pass
    
    async def _screenshot(self) -> Optional[str]:
        try:
            w, h = pyautogui.size()
            with ImageGrab.grab(bbox=(0, 0, w, h)) as snap:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.screenshot_dir, f"screen_{timestamp}.png")
                snap.save(path)
                return path
        except Exception as e:
            print(f"[Screenshot error] {e}")
            return None
    
    async def process_query(self, query: str, image_path: Optional[str] = None) -> Tuple[str, float]:
        start_time = time.time()
        
        # Dynamic task determination
        tasks = self._determine_tasks(query, image_path)
        
        # Execute tasks in parallel
        vision, ocr, context = await asyncio.gather(*tasks)
        
        # # Extract results
        # vision = task_results[0] if 'vision' in tasks[0]._coro.__name__ else AgentResponse("", 0.0, {})
        # ocr = task_results[1] if 'ocr' in tasks[1]._coro.__name__ else AgentResponse("", 0.0, {})
        # context = task_results[2]
        
        # Reasoning with all inputs
        answer = await self.reasoning_agent.reason(query, vision, ocr, context)
        
        # Calculate reward
        reward = self.reward_agent.score(query, answer)
        self.rewards.append(reward)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.query_times.append(processing_time)
        
        # Optional: Update policy based on reward
        if hasattr(self.reasoning_agent.react_optimizer, 'update_policy'):
            self.reasoning_agent.react_optimizer.update_policy(reward)
        
        print(f"[Performance] Time: {processing_time:.2f}s, Reward: {reward:.2f}")
        return answer.content, reward
    
    def _determine_tasks(self, query: str, image_path: Optional[str]) -> List[Coroutine[Any, Any, AgentResponse]]:
        needs_visual = any(k in query.lower() for k in ["diagram", "schematic", "layout", "image"])

        # Default: dummy vision and OCR tasks
        vision_task = asyncio.sleep(0, result=AgentResponse("", 0.0, {}))
        ocr_task = asyncio.sleep(0, result=AgentResponse("", 0.0, {}))


        # """Determine which tasks to run based on query and available inputs"""
        # tasks = []
        
        # # Always include retrieval
        # tasks.append(self.retriever_agent.retrieve(query))
        
        # # Check if we need vision/OCR
        # needs_visual = any(k in query.lower() for k in ["diagram", "schematic", "layout", "image"])
        
        if image_path:
            if needs_visual:
                vision_task = self.vision_agent.analyze(image_path, query)
            ocr_task = self.ocr_agent.extract_text(image_path)
        elif needs_visual:
            # No image provided but needed - capture screenshot
            async def get_image_and_analyze():
                captured_path = await self._screenshot()
                if captured_path:
                    return await self.vision_agent.analyze(captured_path, query)
                return AgentResponse("", 0.0, {})
            
            async def get_image_and_ocr():
                captured_path = await self._screenshot()
                if captured_path:
                    return await self.ocr_agent.extract_text(captured_path)
                return AgentResponse("", 0.0, {})
            
            vision_task = get_image_and_analyze()
            ocr_task = get_image_and_ocr()
        retrieval_task = self.retriever_agent.retrieve(query)

        return [vision_task, ocr_task, retrieval_task]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return current performance metrics"""
        avg_time = sum(self.query_times) / max(1, len(self.query_times))
        avg_reward = sum(self.rewards) / max(1, len(self.rewards))
        
        return {
            "average_query_time": avg_time,
            "average_reward": avg_reward,
            "total_queries": len(self.query_times),
            "recent_rewards": list(self.rewards)[-5:]
        }

async def main():
    config = {
        "text_model": "qwen:1.8b",  # More powerful model
        "vision_model": "qwen:1.8b",  # Vision-language model
        "text_data_dir": "./knowledge_base",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "screenshot_dir": "./screenshots",
        "react_policy_path": "./react_policy.json",
        "reward_model_path": "./reward_model.json"
    }
    
    copilot = ESIMCopilot(config)
    print("🤖 eSim Copilot is ready. Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break
            
            image_path = None
            if user_input.startswith("image:"):
                parts = user_input.split(" ", 1)
                image_path = parts[0][6:].strip()
                user_input = parts[1] if len(parts) > 1 else ""
            
            start_time = time.time()
            response, reward = await copilot.process_query(user_input, image_path)
            elapsed = time.time() - start_time
            
            print(f"\n🧠 Copilot (took {elapsed:.2f}s, confidence: {reward:.2f}):")
            print(response)
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[Error] {e}")
            continue

if __name__ == "__main__":
    import torch  # For GPU detection
    
    asyncio.run(main())