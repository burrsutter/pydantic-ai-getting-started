from typing import Dict, List, Optional
import nest_asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown


nest_asyncio.apply()


model = OpenAIModel("gpt-4o")

# --------------------------------------------------------------
# 2. Agent with Structured Response
# --------------------------------------------------------------
"""
This example shows how to get structured, type-safe responses from the agent.
Key concepts:
- Using Pydantic models to define response structure
- Type validation and safety
- Field descriptions for better model understanding
"""


class ResponseModel(BaseModel):
    """Structured response with metadata."""

    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


agent2 = Agent(
    model=model,
    result_type=ResponseModel,
    system_prompt=(
        "You are an intelligent customer support agent. "
        "Analyze queries carefully and provide structured responses."
    ),
)

response = agent2.run_sync("How can I track my order #12345?")
print(response.data.model_dump_json(indent=2))
