"""
title: CoT-Decoding Pipeline
author: latent-variable
date: 2024-10-15
version: 1.2
license: MIT
description: A pipeline that implements Chain-of-Thought decoding using the Ollama API.
requirements: requests
"""

from typing import List, Union, Generator, Iterator
import os
import requests
import random
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        ollama_api_url: str = "http://localhost:11434/api/generate"
        model: str = ''
        k: int = 10  # Number of top-k alternatives to consider
        temperature: float = 0.7  # Temperature for sampling
        max_tokens: int = 256  # Maximum tokens to generate
        debug: bool = True  # Enable debugging output

    def __init__(self):
        # self.type = "filter"
        self.name = "CoT-Decoding Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "priority": 0,
                "ollama_api_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
                "model": '',
                "k": int(os.getenv("COT_DECODING_K", "10")),
                "temperature": float(os.getenv("COT_DECODING_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("COT_DECODING_MAX_TOKENS", "256")),
                "debug": os.getenv("COT_DECODING_DEBUG", "True") == "True",
            }
        )

    async def on_startup(self):
        print(f"on_startup: {self.name}")

    async def on_shutdown(self):
        print(f"on_shutdown: {self.name}")

    async def on_valves_updated(self):
        # This method can be used to update any configurations if valves are changed
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Implement the CoT-Decoding logic here using the 'pipe' method

        # Format the prompt from the messages in standard QA format
        prompt = self.format_prompt(messages)

        if self.valves.debug:
            print("Formatted Prompt:")
            print(prompt)

        # Get top-k alternative responses using parameters from valves
        top_k_responses = self.get_top_k_responses(
            self.valves.model,
            prompt,
            self.valves.k,
            self.valves.temperature,
            self.valves.max_tokens,
            self.valves.ollama_api_url
        )
        if self.valves.debug:
            print("\nGenerated Responses and Confidence Scores:")
            for idx, response in enumerate(top_k_responses):
                print(f"Response {idx + 1}:")
                print(f"Content: {response['response']}")
                print(f"Confidence: {response.get('confidence', 'Not calculated')}\n")

        # Select the best response based on confidence
        best_response = self.select_best_response(top_k_responses)
        if self.valves.debug:
            print("Selected Best Response:")
            print(f"Content: {best_response['response']}")
            print(f"Confidence: {best_response.get('confidence', 'Not calculated')}")

        if best_response:
            # For debugging purposes, you might want to return all the responses and debug info
            # However, since the pipe function is expected to return a string, we can include the debug info in the return value
            if self.valves.debug:
                debug_info = "\n\n--- Debug Info ---\n"
                for idx, response in enumerate(top_k_responses):
                    debug_info += f"Response {idx + 1}:\n"
                    debug_info += f"Content: {response['response']}\n"
                    debug_info += f"Confidence: {response.get('confidence', 'Not calculated')}\n\n"
                debug_info += f"Selected Response:\nContent: {best_response['response']}\nConfidence: {best_response.get('confidence', 'Not calculated')}\n"
                return best_response["response"] + debug_info
            else:
                return best_response["response"]
        else:
            return "I'm sorry, but I couldn't generate a response."

    def format_prompt(self, messages: List[dict]) -> str:
        # Convert the messages into a single prompt string
        prompt = ""
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                prompt += f"Q: {content}\nA:"
            elif role == "assistant":
                # Optionally include assistant's previous responses
                prompt += f"{content}\n"
        return prompt.strip()

    def get_top_k_responses(
        self, model: str, prompt: str, k: int, temperature: float, max_tokens: int, api_url: str, 
    ) -> List[dict]:
        # Use the Ollama API to get top-k alternative decoding paths  
        responses = []
        for i in range(k):
            seed = random.randint(0, 1e6)
            params = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "seed": seed,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_k": k,
                    "stream": False
                },
                "stream": False
            }
            response = requests.post(api_url, json=params)
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                if response_text:
                    response_info = {
                        "response": response_text,
                        "info": data
                    }
                    # Calculate confidence here to include in debug output
                    confidence = self.calculate_confidence(response_info)
                    response_info["confidence"] = confidence
                    responses.append(response_info)
            else:
                print(f"Error in API call: {response.status_code} - {response.text}")
        return responses

    def select_best_response(self, responses: List[dict]) -> dict:
        # Calculate confidence metrics for each response and select the most reliable one
        best_response = None
        highest_confidence = -1
        for response in responses:
            confidence = response.get("confidence", 0)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_response = response
        return best_response

    def calculate_confidence(self, response: dict) -> float:
        # Implement confidence calculation based on token probabilities
        # For this example, we'll use a simplified confidence metric

        # Retrieve eval_count and eval_duration from the response info
        eval_count = response["info"].get("eval_count", 0)
        eval_duration = response["info"].get("eval_duration", 1)  # Avoid division by zero

        # Confidence metric: tokens per second
        confidence = eval_count / eval_duration

        return confidence