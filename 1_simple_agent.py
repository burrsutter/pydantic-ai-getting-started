from typing import Dict, List, Optional
import nest_asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown


nest_asyncio.apply()


model = OpenAIModel("gpt-4o")

# --------------------------------------------------------------
# 1. Simple Agent - Hello World Example
# --------------------------------------------------------------
"""
This example demonstrates the basic usage of PydanticAI agents.
Key concepts:
- Creating a basic agent with a system prompt
- Running synchronous queries
- Accessing response data, message history, and costs
"""

agent1 = Agent(
    model=model,
    system_prompt="You are a helpful customer support agent. Be concise and friendly.",
)

# Example usage of basic agent
response = agent1.run_sync("How can I track my order #12345?")

print("**************")
print(response.data)
# print(response.all_messages())
# print(response.cost())

print("\n")

response2 = agent1.run_sync(
    user_prompt="What was my previous question?",
    message_history=response.new_messages(),
)
print(response2.data)
print("**************\n")
