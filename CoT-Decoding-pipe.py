"""
title: CoT-Decoding Pipeline with Self-Evaluation
author: latent-variable
date: 2024-10-15
version: 1.6
license: MIT
description: A pipeline that implements Chain-of-Thought decoding using the Ollama API, with self-evaluation and conversation history tracking using messages.
requirements: requests
"""

from typing import List, Union, Generator, Iterator, Optional
import os
import requests
import random
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        ollama_api_url: str = "http://localhost:11434/api/chat"
        model: str = ''
        k: int = 5  # Number of top-k alternatives to consider
        temperature: float = 0.7  # Temperature for sampling
        max_tokens: int = 256  # Maximum tokens to generate
        evaluation_temperature: float = 0.5  # Temperature for the evaluation step
        evaluation_max_tokens: int = 512  # Max tokens for the evaluation response
        debug: bool = True  # Enable debugging output

    def __init__(self):
        self.name = "CoT-Decoding Pipeline with Self-Evaluation"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "priority": 0,
                "ollama_api_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat"),
                "model": '',
                "k": int(os.getenv("COT_DECODING_K", "5")),
                "temperature": float(os.getenv("COT_DECODING_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("COT_DECODING_MAX_TOKENS", "256")),
                "evaluation_temperature": float(os.getenv("EVALUATION_TEMPERATURE", "0.5")),
                "evaluation_max_tokens": int(os.getenv("EVALUATION_MAX_TOKENS", "512")),
                "debug": os.getenv("COT_DECODING_DEBUG", "True") == "True",
            }
        )

    async def on_startup(self):
        print(f"on_startup: {self.name}")

    async def on_shutdown(self):
        print(f"on_shutdown: {self.name}")

    async def on_valves_updated(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.model:
            return f"Please select a model to use with this pipeline {model_id}"

        # Use messages directly
        conversation_history = messages.copy()

        if self.valves.debug:
            print("Conversation History:")
            for msg in conversation_history:
                print(f"{msg['role'].capitalize()}: {msg['content']}")

        # Get top-k alternative responses for the last user message
        last_user_message = self.get_last_user_message(conversation_history)
        if not last_user_message:
            return "No user message found."

        top_k_responses = self.get_top_k_responses(
            self.valves.model or model_id,
            conversation_history,
            self.valves.k,
            self.valves.temperature,
            self.valves.max_tokens,
            self.valves.ollama_api_url
        )

        if self.valves.debug:
            print("\nGenerated Responses:")
            for idx, response in enumerate(top_k_responses):
                print(f"Response {idx + 1}: {response['content']}\n")

        # Use the model to select or generate the best response
        best_response, inner_monologue = self.select_best_response_with_model(
            self.valves.model or model_id,
            conversation_history,
            top_k_responses,
            self.valves.evaluation_temperature,
            self.valves.evaluation_max_tokens,
            self.valves.ollama_api_url
        )

        if self.valves.debug:
            print("\nInner Monologue:")
            print(inner_monologue)
            print("\nSelected Best Response:")
            print(best_response)

        if best_response:
            if self.valves.debug:
                # Include the inner monologue before the final response
                return f"--- Inner Monologue ---\n{inner_monologue}\n\n--- Final Response ---\n{best_response}"
            else:
                return best_response
        else:
            return "I'm sorry, but I couldn't generate a response."

    def get_last_user_message(self, messages: List[dict]) -> Optional[dict]:
        for msg in reversed(messages):
            if msg['role'] == 'user':
                return msg
        return None

    def get_top_k_responses(
        self, model: str, messages: List[dict], k: int, temperature: float, max_tokens: int, api_url: str
    ) -> List[dict]:
        responses = []
        for i in range(k):
            seed = random.randint(0, int(1e6))
            # Prepare the messages for this generation
            messages_for_generation = messages.copy()
            # Remove the assistant's last response if any
            while messages_for_generation and messages_for_generation[-1]['role'] == 'assistant':
                messages_for_generation.pop()

            params = {
                "model": model,
                "messages": messages_for_generation,
                "options": {
                    "seed": seed,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "stream": False
            }
            response = requests.post(api_url, json=params)
            if response.status_code == 200:
                data = response.json()
                assistant_message = data.get("message", {})
                response_text = assistant_message.get("content", "").strip()
                if response_text:
                    responses.append({
                        "content": response_text
                    })
            else:
                print(f"Error in API call: {response.status_code} - {response.text}")
        return responses

    def select_best_response_with_model(
        self, model: str, messages: List[dict], responses: List[dict],
        temperature: float, max_tokens: int, api_url: str
    ) -> (Optional[str], Optional[str]):
        # Prepare the evaluation messages
        evaluation_messages = messages.copy()
        # Remove the assistant's last response if any
        while evaluation_messages and evaluation_messages[-1]['role'] == 'assistant':
            evaluation_messages.pop()

        # Compile the possible responses into a string
        possible_responses_text = "I have considered the following possible responses:\n"
        for idx, response in enumerate(responses, 1):
            possible_responses_text += f"[{idx}] {response['content']}\n"
        possible_responses_text += (
            "\nPlease provide the best possible response to the last user message based on the options above."
            "\nDo not mention the options or this instruction in your final answer."
        )

        # Add the possible responses and instructions as a 'user' message
        evaluation_messages.append({
            "role": "user",
            "content": possible_responses_text
        })

        if self.valves.debug:
            print("\nEvaluation Messages:")
            for msg in evaluation_messages:
                print(f"{msg['role'].capitalize()}: {msg['content']}")

        # Call the model with the evaluation messages
        params = {
            "model": model,
            "messages": evaluation_messages,
            "options": {
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "stream": False
        }
        response = requests.post(api_url, json=params)
        if response.status_code == 200:
            data = response.json()
            assistant_message = data.get("message", {})
            final_response = assistant_message.get("content", "").strip()
            return final_response, possible_responses_text
        else:
            print(f"Error in evaluation API call: {response.status_code} - {response.text}")
            return None, None
