from typing import Dict, List, Optional
import nest_asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown


nest_asyncio.apply()


model = OpenAIModel("gpt-4o")

# --------------------------------------------------------------
# 4. Agent with Tools
# --------------------------------------------------------------

"""
This example shows how to enhance agents with custom tools.
Key concepts:
- Creating and registering tools
- Accessing context in tools
"""
class ResponseModel(BaseModel):
    """Structured response with metadata."""

    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


# Define order schema
class Order(BaseModel):
    """Structure for order details."""

    order_id: str
    status: str
    items: List[str]


# Define customer schema
class CustomerDetails(BaseModel):
    """Structure for incoming customer queries."""

    customer_id: str
    name: str
    email: str
    orders: Optional[List[Order]] = None


customer = CustomerDetails(
    customer_id="1",
    name="John Doe",
    email="john.doe@example.com",
    orders=[
        Order(order_id="12345", status="shipped", items=["Blue Jeans", "T-Shirt"]),
    ],
)


shipping_info_db: Dict[str, str] = {
    "12345": "Shipped on 2024-12-01",
    "67890": "Out for delivery",
}


def get_shipping_info(ctx: RunContext[CustomerDetails]) -> str:
    """Get the customer's shipping information."""
    return shipping_info_db[ctx.deps.orders[0].order_id]


# Agent with structured output and dependencies
agent5 = Agent(
    model=model,
    result_type=ResponseModel,
    deps_type=CustomerDetails,
    retries=3,
    system_prompt=(
        "You are an intelligent customer support agent. "
        "Analyze queries carefully and provide structured responses. "
        "Use tools to look up relevant information."
        "Always great the customer and provide a helpful response."
    ),  # These are known when writing the code
    tools=[Tool(get_shipping_info, takes_ctx=True)],  # Add tool via kwarg
)


@agent5.system_prompt
async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
    return f"Customer details: {to_markdown(ctx.deps)}"


response = agent5.run_sync(
    user_prompt="What's the status of my last order?", deps=customer
)

response.all_messages()
print(response.data.model_dump_json(indent=2))

print(
    "Customer Details:\n"
    f"Name: {customer.name}\n"
    f"Email: {customer.email}\n\n"
    "Response Details:\n"
    f"{response.data.response}\n\n"
    "Status:\n"
    f"Follow-up Required: {response.data.follow_up_required}\n"
    f"Needs Escalation: {response.data.needs_escalation}"
)
