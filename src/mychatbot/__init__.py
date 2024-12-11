from typing import List, Dict, Optional
import requests

class OllamaChat:
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        """Initialize the Ollama chatbot.
        
        Args:
            model: The name of the Ollama model to use
            host: The host URL where Ollama is running
        """
        self.model = model
        self.host = host.rstrip('/')
        self.history: List[Dict[str, str]] = []

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Send a message to the chatbot and get a response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            The model's response as a string
        """
        url = f"{self.host}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": self.history + [{"role": "user", "content": message}],
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            assistant_message = response.json()["message"]["content"]
            
            # Update conversation history
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to communicate with Ollama: {str(e)}")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []


from mychatbot import OllamaChat

# Create a chatbot instance
chatbot = OllamaChat(model="llama2")

# Send a message and get a response
response = chatbot.chat("Hello! How are you?")
print(response)

# Use a system prompt
response = chatbot.chat(
    "Tell me about Python programming.",
    system_prompt="You are a helpful programming assistant."
)
print(response)

# Clear conversation history
chatbot.clear_history()

