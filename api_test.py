from qwen_api import Qwen
from qwen_api.core.types.chat import ChatMessage

client = Qwen()

messages = [ChatMessage(
    role="user",
    content="Solve this step by step: A company's revenue increased by 25% in Q1, decreased by 15% in Q2, and increased by 30% in Q3. If the Q3 revenue is $169,000, what was the initial revenue?",
    web_search=False,
    thinking=True  # Enable thinking mode for step-by-step reasoning
)]

response = client.chat.create(
    messages=messages,
    model="qwen-max-latest"
)

print(response.choices.message.content)