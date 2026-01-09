# Complete LangChain Guide: From Basics to Production AI Agents

## Table of Contents
1. [What is LangChain? The Simple Explanation](#what-is-langchain)
2. [Part 1: LangChain Foundations](#part-1-foundations)
3. [Part 2: Core Components Deep Dive](#part-2-core-components)
4. [Part 3: Building AI Agents with LangChain](#part-3-building-agents)
5. [Part 4: LangChain for Production](#part-4-production)
6. [Part 5: How Minimal AI Uses LangChain](#part-5-minimal-ai)
7. [Part 6: Advanced Patterns](#part-6-advanced-patterns)
8. [Part 7: Complete Project Examples](#part-7-projects)

---

## What is LangChain?

### The Super Simple Explanation (Like You're 10)

**Without LangChain:**
```python
# You have to do EVERYTHING manually
client = Anthropic(api_key="...")
response = client.messages.create(...)
# Parse the response
# Handle tool calls
# Manage conversation history
# Connect to databases
# Handle errors
# 200+ lines of code just to make a simple agent!
```

**With LangChain:**
```python
# LangChain does the hard stuff for you
agent = create_agent(llm, tools)
result = agent.invoke("Cancel order #12345")
# That's it! LangChain handles everything else
```

**LangChain is like LEGO blocks for building AI applications.**

Instead of building everything from scratch, you get pre-made pieces that snap together:
- 🧩 **Models**: Connect to any LLM (Claude, GPT-4, etc.)
- 🛠️ **Tools**: Give your agent superpowers (search web, query database, send emails)
- 💾 **Memory**: Make your agent remember past conversations
- ⛓️ **Chains**: Connect multiple steps together
- 🤖 **Agents**: Build autonomous AI that makes decisions

### Why Companies Like Minimal AI Use LangChain

1. **Speed**: Build in hours, not weeks
2. **Flexibility**: Easy to swap models, add tools, change behavior
3. **Battle-tested**: Used by thousands of companies, bugs already fixed
4. **Ecosystem**: Massive community, tons of integrations
5. **Production-ready**: Built-in monitoring, error handling, retries

**Real Example:**
Building a customer support agent WITHOUT LangChain: 2 weeks
Building the SAME agent WITH LangChain: 2 days

---

## Part 1: LangChain Foundations

### 1.1 Installation and Setup

```bash
# Install LangChain
pip install langchain langchain-anthropic langchain-openai

# Install additional components
pip install langchain-community  # Community integrations
pip install langsmith  # Monitoring and debugging
pip install langgraph  # Advanced agent workflows

# For specific features
pip install faiss-cpu  # Vector database
pip install chromadb  # Another vector database option
pip install pypdf  # PDF processing
pip install beautifulsoup4  # Web scraping
```

**Initial Setup:**

```python
import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Set API keys (do this in .env file in real projects)
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create your first LLM connection
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0
)

# Test it
response = llm.invoke("Say hello!")
print(response.content)  # Output: "Hello! How can I help you today?"
```

### 1.2 The Five Core LangChain Concepts

**Think of building AI like building a car:**

1. **Models** = The engine (ChatAnthropic, ChatOpenAI)
2. **Prompts** = The steering wheel (How you control the AI)
3. **Chains** = The assembly line (Multiple steps working together)
4. **Memory** = The dashboard camera (Remembering what happened)
5. **Agents** = The self-driving system (AI that makes decisions)

Let's understand each one deeply.

---

## Part 2: Core Components Deep Dive

### 2.1 Models (Chat Models)

**Chat Models** = The AI brain you're using

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Different models
claude = ChatAnthropic(model="claude-sonnet-4-5-20250929")
gpt4 = ChatOpenAI(model="gpt-4-turbo")
gpt35 = ChatOpenAI(model="gpt-3.5-turbo")

# Basic usage
response = claude.invoke("What's 2+2?")
print(response.content)  # "2+2 equals 4"

# With messages (full conversation)
messages = [
    SystemMessage(content="You are a helpful math tutor"),
    HumanMessage(content="What's 2+2?"),
]
response = claude.invoke(messages)
```

**Understanding Messages:**

```python
from langchain_core.messages import (
    HumanMessage,      # Messages from the user
    AIMessage,         # Messages from the AI
    SystemMessage,     # System instructions
    FunctionMessage,   # Results from function calls
)

# Building a conversation
conversation = [
    SystemMessage(content="You are a professional translator"),
    HumanMessage(content="Translate 'hello' to Spanish"),
    AIMessage(content="'Hello' in Spanish is 'Hola'"),
    HumanMessage(content="Now translate 'goodbye'"),
]

response = llm.invoke(conversation)
print(response.content)  # "'Goodbye' in Spanish is 'Adiós'"
```

**Streaming Responses:**

```python
# Stream word-by-word (like ChatGPT types)
for chunk in llm.stream("Write me a short poem"):
    print(chunk.content, end="", flush=True)

# Output appears gradually:
# "Roses are red,
# Violets are blue,
# Code is beautiful,
# And so are you."
```

**Async Support (Important for Production):**

```python
import asyncio

async def process_many_requests():
    """Process multiple requests at once"""
    
    tasks = [
        llm.ainvoke("What's 2+2?"),
        llm.ainvoke("What's 3+3?"),
        llm.ainvoke("What's 4+4?"),
    ]
    
    # All run simultaneously!
    results = await asyncio.gather(*tasks)
    return results

# Run it
results = asyncio.run(process_many_requests())
# Takes ~3 seconds instead of 9 seconds (3x faster!)
```

### 2.2 Prompt Templates

**Problem:** Writing prompts manually is tedious and error-prone

```python
# Bad: Manual string formatting
prompt = f"You are a {role}. Help the user with {task}. User said: {message}"
```

**Solution:** Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}"),
    ("human", "{user_input}"),
])

# Use it
prompt = template.invoke({
    "role": "math tutor",
    "user_input": "What's calculus?"
})

response = llm.invoke(prompt)
```

**Advanced Prompt Templates:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template with conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent for {company}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

# Use it
messages = prompt.invoke({
    "company": "ShopCo",
    "chat_history": [
        HumanMessage(content="My order is late"),
        AIMessage(content="I'm sorry to hear that. What's your order number?"),
    ],
    "user_input": "It's order #12345"
})

response = llm.invoke(messages)
```

**Few-Shot Prompting (Teaching by Example):**

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# Examples of what you want
examples = [
    {"input": "I want to cancel", "output": "cancel_order"},
    {"input": "Where's my package?", "output": "track_shipment"},
    {"input": "I need a refund", "output": "process_refund"},
]

# Create few-shot template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Final prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user intent"),
    few_shot_prompt,
    ("human", "{input}"),
])

# Use it
response = llm.invoke(final_prompt.invoke({"input": "Get my money back"}))
# AI learns from examples: likely outputs "process_refund"
```

### 2.3 Output Parsers (Structured Output)

**Problem:** LLM returns text, but you need structured data (JSON, lists, etc.)

```python
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define what structure you want
class CustomerIntent(BaseModel):
    intent: str = Field(description="The customer's intent")
    order_id: str | None = Field(description="Order ID if mentioned")
    urgency: str = Field(description="low, medium, or high")

# Create parser
parser = PydanticOutputParser(pydantic_object=CustomerIntent)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract customer intent.\n{format_instructions}"),
    ("human", "{user_input}"),
])

# Chain it together
chain = prompt | llm | parser

# Use it
result = chain.invoke({
    "user_input": "URGENT! Cancel order #12345 immediately!",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# CustomerIntent(
#     intent="cancel_order",
#     order_id="12345",
#     urgency="high"
# )
```

**Simple List Parser:**

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "List 5 popular programming languages.\n{format_instructions}"),
])

chain = prompt | llm | parser

result = chain.invoke({"format_instructions": parser.get_format_instructions()})
print(result)
# ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

### 2.4 Chains (The Power of LCEL)

**LCEL = LangChain Expression Language**

The magic of LangChain: Connect components with the `|` operator (pipe)

```python
# Simple chain
chain = prompt | llm | parser

# That's equivalent to:
def manual_chain(input):
    formatted_prompt = prompt.invoke(input)
    llm_response = llm.invoke(formatted_prompt)
    parsed_output = parser.parse(llm_response.content)
    return parsed_output
```

**Why LCEL is Powerful:**

```python
# 1. Easy to read
chain = prompt | llm | output_parser

# 2. Automatic streaming
for chunk in chain.stream({"input": "Hello"}):
    print(chunk)

# 3. Automatic async
result = await chain.ainvoke({"input": "Hello"})

# 4. Automatic batching
results = chain.batch([
    {"input": "Hello"},
    {"input": "Hi"},
    {"input": "Hey"},
])

# 5. Easy to modify
new_chain = prompt | different_llm | different_parser
```

**Real Example: Customer Service Classification Chain**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

# Step 1: Classification prompt
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify customer message into categories:
    - order_status
    - cancel_order
    - refund
    - product_question
    - complaint
    - other
    
    Respond with JSON: {{"category": "...", "confidence": 0.0-1.0}}"""),
    ("human", "{message}"),
])

# Step 2: Create chain
classification_chain = (
    classification_prompt 
    | ChatAnthropic(model="claude-sonnet-4-5-20250929")
    | JsonOutputParser()
)

# Use it
result = classification_chain.invoke({
    "message": "Where is my order? It's been 2 weeks!"
})

print(result)
# {'category': 'order_status', 'confidence': 0.95}
```

**Conditional Chains (Branching Logic):**

```python
from langchain_core.runnables import RunnableBranch

# Different chains for different intents
refund_chain = refund_prompt | llm | refund_parser
cancel_chain = cancel_prompt | llm | cancel_parser
general_chain = general_prompt | llm | general_parser

# Route based on classification
def route_chain(input):
    """Decide which chain to use"""
    category = input.get("category")
    
    if category == "refund":
        return refund_chain
    elif category == "cancel_order":
        return cancel_chain
    else:
        return general_chain

# Create routing chain
branch = RunnableBranch(
    (lambda x: x["category"] == "refund", refund_chain),
    (lambda x: x["category"] == "cancel_order", cancel_chain),
    general_chain  # default
)

# Full pipeline
full_chain = classification_chain | branch

# Use it
result = full_chain.invoke({"message": "I want my money back!"})
# Automatically routes to refund_chain
```

### 2.5 Memory (Making Agents Remember)

**Problem:** Each request is independent. The AI forgets everything.

```python
# First message
llm.invoke("My name is Nishant")
# "Nice to meet you, Nishant!"

# Second message
llm.invoke("What's my name?")
# "I don't know your name." ← It forgot!
```

**Solution: Conversation Memory**

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # Shows what's happening
)

# First message
response1 = conversation.predict(input="My name is Nishant")
print(response1)  # "Nice to meet you, Nishant!"

# Second message
response2 = conversation.predict(input="What's my name?")
print(response2)  # "Your name is Nishant!" ← It remembered!

# Check the memory
print(memory.load_memory_variables({}))
# {
#     'history': 'Human: My name is Nishant\nAI: Nice to meet you, Nishant!\nHuman: What's my name?\nAI: Your name is Nishant!'
# }
```

**Different Types of Memory:**

**1. ConversationBufferMemory** (Remembers everything)
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# Stores full conversation
# Pro: Complete history
# Con: Gets expensive with long conversations (many tokens)
```

**2. ConversationBufferWindowMemory** (Last N messages only)
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Only last 5 messages
# Pro: Controlled token usage
# Con: Forgets older context
```

**3. ConversationSummaryMemory** (Summarizes history)
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# Pro: Keeps context without many tokens
# Con: Might lose details in summary
```

**4. ConversationSummaryBufferMemory** (Hybrid approach)
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000  # Summarize if exceeds 1000 tokens
)
# Pro: Recent messages kept, old ones summarized
# Con: More complex
```

**Custom Memory (Most Flexible):**

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

class DatabaseMemory:
    """Store conversation in database"""
    
    def __init__(self, user_id, session):
        self.user_id = user_id
        self.session = session
    
    def save_context(self, inputs, outputs):
        """Save conversation to database"""
        message = {
            "user": inputs.get("input"),
            "ai": outputs.get("output"),
            "timestamp": datetime.now()
        }
        self.session.add(Message(**message, user_id=self.user_id))
        self.session.commit()
    
    def load_memory_variables(self, inputs):
        """Load conversation from database"""
        messages = self.session.query(Message).filter_by(
            user_id=self.user_id
        ).order_by(Message.timestamp.desc()).limit(10).all()
        
        history = "\n".join([
            f"Human: {m.user}\nAI: {m.ai}"
            for m in reversed(messages)
        ])
        
        return {"history": history}
```

### 2.6 Retrievers (Finding Relevant Information)

**Problem:** You have tons of documents. How do you find relevant info?

**Example Use Case:**
- Customer asks: "What's your return policy?"
- You have 1000 pages of documentation
- Need to find the 1-2 relevant paragraphs

**Solution: Vector Embeddings + Retrieval**

**Step 1: Understanding Embeddings**

```python
from langchain_openai import OpenAIEmbeddings

# Create embeddings model
embeddings = OpenAIEmbeddings()

# Convert text to numbers (vector)
text = "The return policy allows returns within 30 days"
vector = embeddings.embed_query(text)

print(len(vector))  # 1536 numbers
print(vector[:5])   # [0.123, -0.456, 0.789, ...]

# These numbers represent the "meaning" of the text
# Similar meanings = similar vectors
```

**Step 2: Store Documents in Vector Database**

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Your documents
documents = [
    Document(page_content="Returns accepted within 30 days", metadata={"source": "policy.pdf"}),
    Document(page_content="Free shipping on orders over $50", metadata={"source": "policy.pdf"}),
    Document(page_content="Customer support: 1-800-SUPPORT", metadata={"source": "contact.pdf"}),
]

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Search for similar documents
results = vectorstore.similarity_search("What's the return window?")

print(results[0].page_content)
# "Returns accepted within 30 days" ← Found it!
```

**Step 3: Use Retriever in Chain**

```python
from langchain.chains import RetrievalQA

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 results

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Put all docs into context
    retriever=retriever
)

# Ask question
response = qa_chain.invoke("Can I return items after 60 days?")
print(response["result"])
# "No, our policy only allows returns within 30 days."
```

**Advanced Retrieval: MultiQueryRetriever**

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Generates multiple search queries from one question
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# User asks: "return policy"
# MultiQueryRetriever generates:
# - "What is the return policy?"
# - "How long do I have to return items?"
# - "What are the conditions for returns?"
# Then searches with all 3 queries and combines results

results = retriever.get_relevant_documents("return policy")
```

### 2.7 Tools (Giving Agents Superpowers)

**Tools** = Functions the AI can call

```python
from langchain_core.tools import tool

# Define a tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: City name
    """
    # In real life: call weather API
    return f"Weather in {location}: 72°F, Sunny"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: Math expression like "2 + 2" or "10 * 5"
    """
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "Invalid expression"

# List of tools
tools = [get_weather, calculate]

# Tool metadata is automatically extracted from docstring
print(get_weather.name)  # "get_weather"
print(get_weather.description)  # "Get the current weather..."
```

**Using Tools with LLM (Function Calling):**

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Ask something that needs a tool
response = llm_with_tools.invoke("What's the weather in Boston?")

# Check if AI wants to use a tool
print(response.tool_calls)
# [
#     {
#         'name': 'get_weather',
#         'args': {'location': 'Boston'},
#         'id': 'call_abc123'
#     }
# ]

# Execute the tool
tool_call = response.tool_calls[0]
tool = next(t for t in tools if t.name == tool_call["name"])
result = tool.invoke(tool_call["args"])

print(result)  # "Weather in Boston: 72°F, Sunny"
```

**Built-in Tools:**

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Web search tool
search = DuckDuckGoSearchRun()
result = search.run("What's the capital of France?")

# Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wikipedia.run("Python programming language")

# Many more built-in tools:
# - GoogleSearchAPIWrapper
# - PythonREPLTool (run Python code)
# - ShellTool (run shell commands)
# - SQLDatabaseToolkit (query databases)
# - And hundreds more!
```

**Custom Tool for Real Use Case:**

```python
from langchain_core.tools import tool
import requests

@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of an order.
    
    Args:
        order_id: The order ID (e.g., ORD-12345)
    """
    # Mock implementation - replace with real API
    return f"Order {order_id}: Status is 'Shipped'. Expected delivery: Jan 10"

@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a customer order.
    
    Args:
        order_id: Order ID to cancel
        reason: Reason for cancellation
    """
    # Mock implementation
    return f"Order {order_id} cancelled successfully. Refund will process in 3-5 business days."

@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for an order.
    
    Args:
        order_id: Order ID to refund
        amount: Refund amount
        reason: Reason for refund
    """
    return f"Refund of ${amount} processed for order {order_id}. Transaction ID: REF-789"

@tool
def check_inventory(product_id: str) -> str:
    """Check if a product is in stock.
    
    Args:
        product_id: Product ID to check
    """
    return f"Product {product_id}: 25 units in stock. Available for immediate shipping."

@tool
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for information.
    
    Args:
        query: Search query
    """
    # This would use RAG in production
    retriever = knowledge_base.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs[:3]])

tools = [
    get_order_status,
    cancel_order,
    process_refund,
    check_inventory,
    search_knowledge_base
]

# ========== KNOWLEDGE BASE (RAG) ==========

def setup_knowledge_base():
    """Initialize knowledge base with company docs"""
    
    from langchain_core.documents import Document
    
    # Your company documentation
    docs = [
        Document(
            page_content="Return policy: Items can be returned within 30 days of purchase for a full refund.",
            metadata={"source": "policy.pdf", "page": 1}
        ),
        Document(
            page_content="Shipping: Free shipping on orders over $50. Standard shipping takes 5-7 business days.",
            metadata={"source": "shipping.pdf", "page": 1}
        ),
        Document(
            page_content="Refunds are processed within 3-5 business days after we receive the returned item.",
            metadata={"source": "refund.pdf", "page": 1}
        ),
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

knowledge_base = setup_knowledge_base()

# ========== AGENT SETUP ==========

class CustomerSupportAgent:
    """Production-ready customer support agent"""
    
    def __init__(self, store_id: str):
        self.store_id = store_id
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0
        )
        
        # Custom prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional customer service agent for ShopCo.

Your capabilities:
- Check order status
- Cancel orders (within 48 hours of purchase)
- Process refunds (within 30 days)
- Check product availability
- Answer policy questions using knowledge base

Your personality:
- Professional but warm and friendly
- Empathetic to customer concerns
- Clear and concise in explanations
- Proactive in offering solutions

Important rules:
- Always verify order ownership before taking actions
- Follow company policies strictly
- If you can't help, escalate to human agent
- Never make promises you can't keep

Current date: {current_date}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, tools, self.prompt)
        
        # Create executor
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def handle_message(
        self,
        user_id: str,
        message: str,
        session_id: str
    ) -> dict:
        """Handle a customer message"""
        
        from datetime import datetime
        
        # Load conversation history
        history = PostgresChatMessageHistory(
            connection_string=os.getenv("DATABASE_URL"),
            session_id=session_id
        )
        
        # Invoke agent
        result = self.executor.invoke({
            "input": message,
            "chat_history": history.messages,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        })
        
        # Save to history
        history.add_user_message(message)
        history.add_ai_message(result["output"])
        
        # Log interaction (for evaluation)
        self._log_interaction(user_id, session_id, message, result)
        
        return {
            "response": result["output"],
            "intermediate_steps": result.get("intermediate_steps", [])
        }
    
    def _log_interaction(self, user_id: str, session_id: str, message: str, result: dict):
        """Log for monitoring and evaluation"""
        
        # In production: send to LangSmith, database, etc.
        log_data = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "session_id": session_id,
            "user_message": message,
            "agent_response": result["output"],
            "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
            "store_id": self.store_id
        }
        
        # Save to database
        # save_to_analytics_db(log_data)

# ========== USAGE ==========

# Create agent
agent = CustomerSupportAgent(store_id="store-123")

# Handle customer message
response = agent.handle_message(
    user_id="user-456",
    message="I want to return order #12345",
    session_id="session-789"
)

print(response["response"])
```

### 7.2 Project: Self-Improving Agent System

```python
"""
Complete self-improving agent system
Agents get better over time automatically
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import json

class SelfImprovingAgent:
    """Agent that improves based on feedback"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        self.supervisor_llm = ChatAnthropic(model="claude-opus-4")
        
        # Load or create base prompt
        self.prompt_version = self._load_prompt_version()
        self.interactions_db = []
        self.feedback_db = []
    
    def _load_prompt_version(self) -> str:
        """Load current prompt version"""
        # In production: load from database
        return """You are a customer service agent.
        
Be professional, empathetic, and helpful.
Follow company policies.
"""
    
    def handle_interaction(self, user_message: str) -> dict:
        """Process customer message"""
        
        # Create prompt with current version
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_version),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"input": user_message})
        
        # Store interaction
        interaction = {
            "id": len(self.interactions_db),
            "timestamp": datetime.now(),
            "user_message": user_message,
            "agent_response": response.content,
            "prompt_version": self.prompt_version,
            "feedback": None  # Will be added later
        }
        
        self.interactions_db.append(interaction)
        
        return {
            "response": response.content,
            "interaction_id": interaction["id"]
        }
    
    def record_feedback(
        self,
        interaction_id: int,
        feedback_type: str,
        feedback_data: dict
    ):
        """Record feedback on an interaction"""
        
        feedback = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now(),
            "type": feedback_type,  # "user_rating", "correction", "escalation"
            "data": feedback_data
        }
        
        self.feedback_db.append(feedback)
        self.interactions_db[interaction_id]["feedback"] = feedback
        
        # Check if we should improve
        if len(self.feedback_db) >= 10:
            self._analyze_and_improve()
    
    def _analyze_and_improve(self):
        """Analyze feedback and improve prompt"""
        
        # Get recent negative feedback
        recent_issues = [
            f for f in self.feedback_db[-20:]
            if f["data"].get("rating", 5) < 3 or f["type"] == "escalation"
        ]
        
        if len(recent_issues) < 3:
            return  # Not enough issues to improve
        
        # Analyze patterns
        analysis_prompt = f"""
        Analyze these customer service interactions that received negative feedback:
        
        {self._format_issues(recent_issues)}
        
        Current agent prompt:
        {self.prompt_version}
        
        Identify:
        1. Common patterns in failures
        2. What the agent is doing wrong
        3. Specific improvements needed
        
        Respond in JSON:
        {{
            "patterns": ["pattern1", "pattern2"],
            "root_causes": ["cause1", "cause2"],
            "recommendations": ["rec1", "rec2"]
        }}
        """
        
        analysis = self.supervisor_llm.invoke(analysis_prompt)
        analysis_data = json.loads(analysis.content)
        
        # Generate improved prompt
        new_prompt = self._generate_improved_prompt(analysis_data)
        
        # A/B test new prompt
        self._start_ab_test(self.prompt_version, new_prompt)
    
    def _generate_improved_prompt(self, analysis: dict) -> str:
        """Generate improved prompt based on analysis"""
        
        improvement_prompt = f"""
        Current prompt:
        {self.prompt_version}
        
        Analysis of issues:
        - Patterns: {analysis['patterns']}
        - Root causes: {analysis['root_causes']}
        - Recommendations: {analysis['recommendations']}
        
        Generate an improved version that:
        1. Addresses the identified issues
        2. Maintains current strengths
        3. Adds specific guidance to prevent failures
        
        Return ONLY the new prompt text, no explanation.
        """
        
        response = self.supervisor_llm.invoke(improvement_prompt)
        return response.content
    
    def _format_issues(self, issues: list) -> str:
        """Format issues for analysis"""
        
        formatted = []
        for issue in issues:
            interaction = self.interactions_db[issue["interaction_id"]]
            formatted.append(f"""
Issue {issue['interaction_id']}:
User: {interaction['user_message']}
Agent: {interaction['agent_response']}
Feedback: {issue['data']}
---
""")
        return "\n".join(formatted)
    
    def _start_ab_test(self, prompt_a: str, prompt_b: str):
        """A/B test two prompt versions"""
        
        print(f"Starting A/B test...")
        print(f"Prompt A (current): {len(prompt_a)} chars")
        print(f"Prompt B (new): {len(prompt_b)} chars")
        
        # In production: 
        # - Route 50% traffic to each
        # - Track metrics for 1 week
        # - Pick winner automatically
        # - Roll out to 100%

# ========== USER CORRECTION SYSTEM ==========

class UserCorrectionSystem:
    """Allow users to correct agent responses"""
    
    def __init__(self, agent: SelfImprovingAgent):
        self.agent = agent
        self.corrections = []
    
    def submit_correction(
        self,
        interaction_id: int,
        wrong_response: str,
        correct_response: str,
        explanation: str
    ):
        """User corrects agent's response"""
        
        correction = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now(),
            "wrong": wrong_response,
            "correct": correct_response,
            "explanation": explanation
        }
        
        self.corrections.append(correction)
        
        # Record as feedback
        self.agent.record_feedback(
            interaction_id,
            "correction",
            correction
        )
        
        # After 5 corrections, update prompt immediately
        if len(self.corrections) >= 5:
            self._quick_improve_from_corrections()
    
    def _quick_improve_from_corrections(self):
        """Quickly improve based on corrections"""
        
        recent = self.corrections[-5:]
        
        improvement_prompt = f"""
        The agent made these mistakes (corrected by user):
        
        {self._format_corrections(recent)}
        
        Current prompt:
        {self.agent.prompt_version}
        
        Add specific guidance to prevent these mistakes.
        Return updated prompt.
        """
        
        new_prompt = self.agent.supervisor_llm.invoke(improvement_prompt)
        
        # Update prompt immediately (no A/B test for user corrections)
        self.agent.prompt_version = new_prompt.content
        
        print("✓ Agent improved based on your corrections!")
    
    def _format_corrections(self, corrections: list) -> str:
        formatted = []
        for c in corrections:
            interaction = self.agent.interactions_db[c["interaction_id"]]
            formatted.append(f"""
Situation: {interaction['user_message']}
Agent said (wrong): {c['wrong']}
Should say (correct): {c['correct']}
Why: {c['explanation']}
---
""")
        return "\n".join(formatted)

# ========== USAGE EXAMPLE ==========

# Create self-improving agent
agent = SelfImprovingAgent(agent_id="agent-001")

# User correction system
correction_system = UserCorrectionSystem(agent)

# Customer interaction
response1 = agent.handle_interaction("I want a refund for order #12345")
print(response1["response"])

# User corrects it
correction_system.submit_correction(
    interaction_id=response1["interaction_id"],
    wrong_response=response1["response"],
    correct_response="I'd be happy to help with that refund! Let me check your order #12345...",
    explanation="Should be more empathetic and action-oriented"
)

# After 5 corrections, agent automatically improves!
```

### 7.3 Comparison: With vs Without LangChain

**Building Customer Support Agent WITHOUT LangChain:**

```python
# ~500+ lines of code

import anthropic
import json
from typing import List, Dict

class ManualAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.conversation_history = {}
        
    def process(self, user_id: str, message: str):
        # Manually manage conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Manually format messages
        messages = self._format_messages(user_id, message)
        
        # Manually define tools
        tools = self._get_tool_definitions()
        
        # Agent loop - ALL MANUAL
        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                tools=tools,
                messages=messages
            )
            
            # Manually handle tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        # Manually execute tool
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                
                # Manually add to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            elif response.stop_reason == "end_turn":
                # Extract final response
                final = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final += block.text
                
                # Manually save history
                self.conversation_history[user_id].append({
                    "role": "user",
                    "content": message
                })
                self.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": final
                })
                
                return final
    
    def _format_messages(self, user_id: str, message: str):
        # Manually format conversation
        messages = []
        
        # Add history
        for msg in self.conversation_history.get(user_id, []):
            messages.append(msg)
        
        # Add new message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _get_tool_definitions(self):
        # Manually define every tool
        return [
            {
                "name": "get_order_status",
                "description": "Get order status",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"}
                    },
                    "required": ["order_id"]
                }
            },
            # ... define 10+ more tools manually
        ]
    
    def _execute_tool(self, tool_name: str, tool_input: dict):
        # Manually route to correct function
        if tool_name == "get_order_status":
            return self.get_order_status(**tool_input)
        elif tool_name == "cancel_order":
            return self.cancel_order(**tool_input)
        # ... handle 10+ more tools
    
    # Need to manually implement:
    # - Error handling
    # - Retries
    # - Streaming
    # - Async support
    # - Logging
    # - Monitoring
    # - Caching
    # - Rate limiting
    # ... 300+ more lines
```

**Same Agent WITH LangChain:**

```python
# ~50 lines of code

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.memory import PostgresChatMessageHistory

# Define tools (automatic schema generation)
@tool
def get_order_status(order_id: str) -> str:
    """Get order status"""
    return f"Order {order_id}: Shipped"

@tool
def cancel_order(order_id: str) -> str:
    """Cancel order"""
    return f"Order {order_id}: Cancelled"

# Create agent (everything automatic)
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
tools = [get_order_status, cancel_order]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=PostgresChatMessageHistory(connection_string="...")
)

# Use it (automatic everything!)
result = executor.invoke({"input": "Cancel order #12345"})
print(result["output"])

# LangChain handles:
# ✓ Tool execution
# ✓ Agent loop
# ✓ Conversation history
# ✓ Error handling
# ✓ Streaming (executor.stream())
# ✓ Async (executor.ainvoke())
# ✓ Monitoring (with LangSmith)
# ✓ Retries
# ✓ All the hard stuff!
```

**That's the power of LangChain: 10x less code, 10x faster development.**

---

## Summary: Key Takeaways

### What is LangChain?
- **Framework** for building LLM applications
- **Pre-built components** that work together
- **Battle-tested** by thousands of companies
- **Production-ready** with monitoring, error handling, etc.

### Core Concepts:
1. **Models** - Connect to any LLM (Claude, GPT-4, etc.)
2. **Prompts** - Template system for consistent prompts
3. **Chains** - Connect steps with `|` operator (LCEL)
4. **Memory** - Make agents remember conversations
5. **Tools** - Give agents superpowers (functions they can call)
6. **Agents** - AI that makes decisions and takes actions
7. **Retrievers** - Find relevant info from documents (RAG)

### Why Minimal AI Uses LangChain:
- **Build faster**: Agents in days, not weeks
- **Scale easier**: Built-in async, streaming, batching
- **Iterate quickly**: Swap models, prompts, tools easily
- **Monitor everything**: LangSmith for debugging
- **Production-ready**: Error handling, retries, fallbacks

### Best Practices:
1. **Start simple** - Get basic agent working first
2. **Add complexity gradually** - Don't over-engineer
3. **Monitor everything** - Use LangSmith from day 1
4. **Test thoroughly** - LLMs are non-deterministic
5. **Handle errors gracefully** - Agents will fail sometimes
6. **Iterate based on data** - Let metrics guide improvements
7. **Use async for scale** - Critical for production
8. **Cache aggressively** - Saves money and latency

### Your Learning Path:

**Week 1-2: Foundations**
- [ ] Install LangChain and dependencies
- [ ] Build basic chatbot with memory
- [ ] Create 3+ tools
- [ ] Build simple agent

**Week 3-4: Intermediate**
- [ ] Add RAG for knowledge base
- [ ] Build multi-step agent workflows
- [ ] Implement error handling
- [ ] Add LangSmith monitoring

**Week 5-6: Advanced**
- [ ] Build multi-agent system
- [ ] Implement self-improvement loop
- [ ] Add async/streaming
- [ ] Deploy to production

**Week 7-8: Polish**
- [ ] Build complete portfolio project
- [ ] Add comprehensive tests
- [ ] Write documentation
- [ ] Deploy with Docker

### Interview Prep:

**Key Questions to Prepare:**

1. **"What is LangChain and why use it?"**
   - "LangChain is a framework that provides pre-built components for building LLM applications. Instead of writing hundreds of lines to handle agent loops, tool calling, memory, and error handling, LangChain provides abstractions that work out of the box. This lets you build in days instead of weeks."

2. **"Explain the agent loop in LangChain"**
   - "An agent repeatedly calls the LLM with available tools, executes any tools the LLM requests, feeds results back, and continues until the task is complete. LangChain's AgentExecutor handles this loop automatically with built-in error handling and iteration limits."

3. **"How do you handle agent errors in production?"**
   - "Multiple strategies: set max_iterations to prevent infinite loops, implement fallback chains for when primary fails, use circuit breakers for failing services, comprehensive logging with LangSmith, and graceful degradation to human handoff when agent can't complete the task."

4. **"What's the difference between a Chain and an Agent?"**
   - "A Chain executes a fixed sequence of steps. An Agent decides what actions to take dynamically based on the situation. Chains are deterministic and predictable; Agents are flexible but less predictable."

5. **"How would you scale an agent to handle 10,000 requests/day?"**
   - "Use async agent executors, implement request queuing with Celery, cache responses aggressively, use rate limiting per user, deploy multiple agent workers behind a load balancer, and monitor performance with LangSmith to identify bottlenecks."

### Resources:

**Official:**
- LangChain Docs: https://python.langchain.com/docs/
- LangSmith: https://smith.langchain.com/
- LangChain GitHub: https://github.com/langchain-ai/langchain

**Learning:**
- LangChain YouTube Channel
- DeepLearning.AI LangChain Courses
- LangChain Cookbook (examples)

**Community:**
- LangChain Discord
- r/LangChain on Reddit
- Twitter: @LangChainAI

---

## Final Thoughts

LangChain is **the industry standard** for building AI agents. Understanding it deeply gives you:

1. **Speed** - Build 10x faster than from scratch
2. **Best Practices** - Learn from thousands of companies
3. **Job Market** - Most AI companies use LangChain
4. **Flexibility** - Easy to customize and extend
5. **Community** - Huge ecosystem and support

For Minimal AI specifically, LangChain enables them to:
- Build agents that integrate with any e-commerce platform
- Let non-technical users configure agents (self-serve)
- Build the "AI Manager" that improves other AIs
- Scale to thousands of stores efficiently
- Iterate quickly based on customer feedback

**Your competitive advantage**: Most candidates know basic LLM usage. Showing deep LangChain knowledge puts you in the top 10%.

Now go build! 🚀
    """Get the status of a customer order.
    
    Args:
        order_id: The order ID (e.g., ORD-12345)
    """
    # Call your API
    response = requests.get(f"https://api.yourcompany.com/orders/{order_id}")
    
    if response.status_code == 200:
        data = response.json()
        return f"Order {order_id}: {data['status']}"
    else:
        return f"Order {order_id} not found"

@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a customer order.
    
    Args:
        order_id: The order ID to cancel
        reason: Reason for cancellation
    """
    response = requests.post(
        f"https://api.yourcompany.com/orders/{order_id}/cancel",
        json={"reason": reason}
    )
    
    if response.status_code == 200:
        return f"Order {order_id} cancelled successfully"
    else:
        return f"Failed to cancel order {order_id}"
```

---

## Part 3: Building AI Agents with LangChain

### 3.1 What Makes an Agent Different?

**Chain:** Fixed sequence of steps
```
User asks question → Search docs → Generate answer
(Always the same steps)
```

**Agent:** Decides what to do
```
User asks question → Agent thinks → "I should search docs"
                                  → Searches docs
                                  → Agent thinks → "I need more info"
                                  → Searches web
                                  → Agent thinks → "Now I can answer"
                                  → Generates answer
```

**Key Difference:** Agent chooses its own actions based on the situation

### 3.2 Creating Your First Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

# Define tools
@tool
def get_order_status(order_id: str) -> str:
    """Get order status"""
    return f"Order {order_id}: Shipped"

@tool  
def cancel_order(order_id: str) -> str:
    """Cancel an order"""
    return f"Order {order_id}: Cancelled"

tools = [get_order_status, cancel_order]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Help users with their orders."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Agent's thinking
])

# Create LLM
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create executor (runs the agent)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # See what agent is thinking
)

# Use it!
result = agent_executor.invoke({
    "input": "I want to cancel order #12345"
})

print(result["output"])
```

**What Happens Behind the Scenes:**

```
1. User: "I want to cancel order #12345"

2. Agent thinks:
   "The user wants to cancel an order. I have a cancel_order tool.
    I should use it with order_id='12345'"

3. Agent calls: cancel_order(order_id="12345")

4. Tool returns: "Order 12345: Cancelled"

5. Agent thinks:
   "The order was cancelled successfully. I should tell the user."

6. Agent responds: "I've cancelled your order #12345 for you."
```

### 3.3 Agent Types in LangChain

**1. Tool Calling Agent (Modern, Best)**
```python
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)
# Uses native function calling (most reliable)
# Works with Claude, GPT-4, etc.
```

**2. OpenAI Functions Agent**
```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
# Specifically for OpenAI models
# Uses function calling API
```

**3. ReAct Agent** (Reason + Act)
```python
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
# Agent explicitly reasons about what to do
# Good for transparency (can see thinking)
```

**4. Structured Chat Agent**
```python
from langchain.agents import create_structured_chat_agent

agent = create_structured_chat_agent(llm, tools, prompt)
# For tools with complex inputs
```

### 3.4 Multi-Step Agent Example

```python
from langchain_core.tools import tool

# Define multiple tools
@tool
def check_inventory(product_id: str) -> str:
    """Check if product is in stock"""
    # Fake implementation
    return f"Product {product_id}: 15 units in stock"

@tool
def get_order(order_id: str) -> str:
    """Get order details"""
    return f"Order {order_id}: 2x Product-ABC, Status: Processing"

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order"""
    return f"Order {order_id} cancelled"

@tool
def notify_customer(message: str) -> str:
    """Send notification to customer"""
    return f"Notification sent: {message}"

tools = [check_inventory, get_order, cancel_order, notify_customer]

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Complex request
result = agent_executor.invoke({
    "input": """
    Customer wants to cancel order #12345 and buy product XYZ instead.
    Check if XYZ is in stock first, then cancel the old order and notify them.
    """
})

# Agent automatically:
# 1. Calls check_inventory("XYZ")
# 2. Calls get_order("12345") 
# 3. Calls cancel_order("12345")
# 4. Calls notify_customer("Your order is cancelled. XYZ is in stock!")
```

### 3.5 Agent with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Update prompt to include history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent with memory
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
response1 = agent_executor.invoke({"input": "My name is Nishant"})
# "Nice to meet you, Nishant!"

response2 = agent_executor.invoke({"input": "What's my name?"})
# "Your name is Nishant!" ← Remembered!

response3 = agent_executor.invoke({"input": "Check order #12345"})
# Uses tools AND remembers context
```

### 3.6 Agent Error Handling

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,  # Prevent infinite loops
    max_execution_time=30,  # 30 second timeout
    early_stopping_method="generate",  # How to stop if max reached
    handle_parsing_errors=True,  # Handle malformed tool calls
)

# With error handling
try:
    result = agent_executor.invoke({"input": "Cancel all orders"})
except Exception as e:
    print(f"Agent error: {e}")
    # Fallback to human agent
```

### 3.7 LangGraph: Advanced Agent Workflows

**LangGraph** = Build agents as graphs (more control than basic agents)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage

# Define state
class AgentState(TypedDict):
    messages: list
    current_step: str
    order_id: str | None

# Define nodes (steps)
def classify_intent(state: AgentState):
    """Figure out what user wants"""
    last_message = state["messages"][-1].content
    
    # Use LLM to classify
    response = llm.invoke(f"Classify intent: {last_message}")
    
    if "cancel" in response.content.lower():
        return {"current_step": "cancel"}
    else:
        return {"current_step": "status"}

def handle_cancellation(state: AgentState):
    """Cancel order"""
    order_id = extract_order_id(state["messages"][-1].content)
    result = cancel_order(order_id)
    
    return {
        "messages": state["messages"] + [AIMessage(content=result)],
        "current_step": "complete"
    }

def handle_status_check(state: AgentState):
    """Check status"""
    order_id = extract_order_id(state["messages"][-1].content)
    result = get_order_status(order_id)
    
    return {
        "messages": state["messages"] + [AIMessage(content=result)],
        "current_step": "complete"
    }

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("cancel", handle_cancellation)
workflow.add_node("status", handle_status_check)

# Add edges (connections)
workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    lambda state: state["current_step"],
    {
        "cancel": "cancel",
        "status": "status"
    }
)

workflow.add_edge("cancel", END)
workflow.add_edge("status", END)

# Compile
app = workflow.compile()

# Use it
result = app.invoke({
    "messages": [HumanMessage(content="Cancel order #12345")],
    "current_step": "start",
    "order_id": None
})
```

**Why LangGraph:**
- More control over agent flow
- Conditional branching
- Parallel execution
- State management
- Error recovery

---

## Part 4: LangChain for Production

### 4.1 LangSmith (Monitoring & Debugging)

**LangSmith** = See exactly what your agent is doing

```python
import os

# Enable LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "customer-support-agent"

# Now all chains/agents are automatically logged!
agent_executor.invoke({"input": "Cancel order #12345"})

# Go to smith.langchain.com to see:
# - Full conversation
# - Every tool call
# - Time taken for each step
# - Tokens used
# - Errors
```

**Adding Custom Metadata:**

```python
from langsmith import traceable

@traceable(
    run_type="chain",
    name="order_cancellation",
    metadata={"version": "1.0", "environment": "production"}
)
def cancel_order_flow(order_id: str):
    # Your logic
    pass
```

### 4.2 Streaming for Better UX

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create agent with streaming
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StreamingStdOutCallbackHandler()],  # Stream to stdout
    verbose=True
)

# For web apps, use custom callback
from langchain.callbacks.base import BaseCallbackHandler

class WebSocketCallback(BaseCallbackHandler):
    """Stream to websocket"""
    
    def __init__(self, websocket):
        self.websocket = websocket
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Called for each token generated"""
        self.websocket.send(token)

# Use it
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[WebSocketCallback(websocket)]
)
```

### 4.3 Async Agents for Scale

```python
import asyncio
from langchain.agents import AgentExecutor

# Create async agent executor
async_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

async def handle_multiple_users():
    """Handle many requests at once"""
    
    tasks = [
        async_agent_executor.ainvoke({"input": "Cancel order #1"}),
        async_agent_executor.ainvoke({"input": "Cancel order #2"}),
        async_agent_executor.ainvoke({"input": "Cancel order #3"}),
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Run it
results = asyncio.run(handle_multiple_users())
```

### 4.4 Caching for Cost Reduction

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

# First call: Hits API ($$$)
response1 = llm.invoke("What's 2+2?")

# Second call: Uses cache (FREE!)
response2 = llm.invoke("What's 2+2?")

# Use Redis for production
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(host='localhost', port=6379)
set_llm_cache(RedisCache(redis_client))
```

### 4.5 Rate Limiting

```python
from langchain.llms.base import BaseLLM
from langchain_anthropic import ChatAnthropic
import time
from collections import deque

class RateLimitedLLM(BaseLLM):
    """LLM with rate limiting"""
    
    def __init__(self, llm, max_requests_per_minute=60):
        self.llm = llm
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = deque()
    
    def _call(self, prompt, stop=None):
        # Check rate limit
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
        
        # Check if at limit
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            print(f"Rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        # Record this request
        self.request_times.append(now)
        
        # Make actual call
        return self.llm._call(prompt, stop)

# Use it
rate_limited_llm = RateLimitedLLM(
    llm=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    max_requests_per_minute=10
)
```

### 4.6 Error Recovery

```python
from langchain.callbacks.base import BaseCallbackHandler
import logging

class ErrorRecoveryCallback(BaseCallbackHandler):
    """Handle errors gracefully"""
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when tool fails"""
        logging.error(f"Tool error: {error}")
        
        # Try alternative tool or fallback
        return "I encountered an error. Let me try a different approach."
    
    def on_agent_error(self, error: Exception, **kwargs):
        """Called when agent fails"""
        logging.error(f"Agent error: {error}")
        
        # Escalate to human
        return "I'm having trouble with this request. Let me connect you with a human agent."

# Use it
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[ErrorRecoveryCallback()],
    handle_parsing_errors=True
)
```

---

## Part 5: How Minimal AI Uses LangChain

### 5.1 Minimal AI's Architecture (Educated Inference)

Based on what they do (autonomous customer support), here's likely how they use LangChain:

```python
# Simplified version of Minimal AI's probable architecture

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain.memory import PostgresChatMessageHistory
from langgraph.graph import StateGraph

# 1. TOOLS: E-commerce integrations
@tool
def cancel_order(order_id: str, store_id: str) -> dict:
    """Cancel customer order in Shopify/WooCommerce/etc"""
    # Call store's API
    pass

@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Process refund"""
    pass

@tool
def check_inventory(product_id: str, store_id: str) -> dict:
    """Check product availability"""
    pass

@tool
def update_order_status(order_id: str, status: str) -> dict:
    """Update order status"""
    pass

@tool
def send_email(to: str, template: str, data: dict) -> dict:
    """Send customer email"""
    pass

@tool
def search_knowledge_base(query: str) -> str:
    """Search company's help docs"""
    # RAG with vector store
    pass

# 2. AGENT: Customer support agent
class SupportAgent:
    def __init__(self, store_id: str):
        self.store_id = store_id
        self.llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        self.tools = [cancel_order, issue_refund, check_inventory, 
                      update_order_status, send_email, search_knowledge_base]
        
        # Custom prompt for this store
        self.prompt = self._load_store_prompt(store_id)
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools)
    
    def handle_ticket(self, ticket_id: str, message: str):
        """Handle support ticket"""
        
        # Load conversation history
        memory = PostgresChatMessageHistory(
            connection_string="...",
            session_id=ticket_id
        )
        
        # Process with agent
        result = self.executor.invoke({
            "input": message,
            "chat_history": memory.messages,
            "store_id": self.store_id
        })
        
        # Save to memory
        memory.add_user_message(message)
        memory.add_ai_message(result["output"])
        
        return result

# 3. AI MANAGER: Monitors and improves agents
class AIManager:
    """The 'AI managing AI' system"""
    
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-opus-4")  # Stronger model
    
    def evaluate_interaction(self, ticket_id: str):
        """Evaluate how agent handled ticket"""
        
        # Get conversation
        conversation = get_conversation(ticket_id)
        
        # Evaluate with LLM
        eval_prompt = f"""
        Evaluate this customer support interaction:
        
        {conversation}
        
        Rate on:
        - Customer satisfaction (1-10)
        - Issue resolved? (yes/no)
        - Response time (fast/medium/slow)
        - Tone (professional/friendly/too casual/rude)
        
        Suggest improvements.
        """
        
        evaluation = self.llm.invoke(eval_prompt)
        
        # Store evaluation
        save_evaluation(ticket_id, evaluation)
        
        # If bad rating, add to training data
        if evaluation.rating < 5:
            self.add_to_training(ticket_id, evaluation)
    
    def generate_training_feedback(self, agent_id: str):
        """Generate feedback for agent improvement"""
        
        # Analyze recent interactions
        interactions = get_recent_interactions(agent_id)
        
        # Find patterns
        prompt = f"""
        Analyze these interactions and suggest improvements:
        
        {interactions}
        
        Common mistakes:
        Areas for improvement:
        Examples of good responses:
        """
        
        feedback = self.llm.invoke(prompt)
        
        # Update agent's system prompt with feedback
        update_agent_prompt(agent_id, feedback)

# 4. SELF-SERVE SETUP: Let customers configure agents
class AgentConfigurator:
    """UI for non-technical users to set up agents"""
    
    def train_from_examples(self, store_id: str, examples: list):
        """Learn from examples user provides"""
        
        # Examples: [{"input": "angry customer", "desired_output": "empathetic response"}]
        
        # Generate system prompt from examples
        prompt_template = """
        You are a customer service agent for {company}.
        
        Based on these examples of how we handle situations:
        {examples}
        
        Apply these principles:
        {extracted_principles}
        """
        
        # Use LLM to extract principles
        principles = self.llm.invoke(f"Extract principles from: {examples}")
        
        # Create agent with custom prompt
        agent = create_custom_agent(store_id, prompt_template.format(
            company=get_store_name(store_id),
            examples=examples,
            extracted_principles=principles
        ))
        
        return agent

# 5. API INTEGRATION AGENT: Explores unfamiliar APIs
class APIExplorerAgent:
    """Figures out how to integrate with new APIs"""
    
    def explore_api(self, api_docs_url: str):
        """Learn how to use an API"""
        
        # Fetch docs
        docs = fetch_and_parse_docs(api_docs_url)
        
        # Use agent to understand API
        explorer_prompt = """
        You are an API integration expert.
        
        Given these API docs:
        {docs}
        
        Figure out:
        1. How to authenticate
        2. How to get order data
        3. How to cancel orders
        4. How to process refunds
        
        Generate Python code to interact with this API.
        """
        
        # Agent generates integration code
        integration_code = self.llm.invoke(explorer_prompt.format(docs=docs))
        
        # Test the generated code
        test_result = self.test_integration(integration_code)
        
        if test_result.success:
            return integration_code
        else:
            # Agent iterates and fixes issues
            return self.fix_integration(integration_code, test_result.error)
```

### 5.2 Key LangChain Patterns Minimal AI Likely Uses

**1. ReAct Agents for Transparency**
```python
from langchain.agents import create_react_agent

# ReAct = Reason + Act
# Agent explicitly shows its thinking
# Helps with debugging and trust

agent = create_react_agent(llm, tools, prompt)

# Output:
# Thought: The user wants to cancel order #12345
# Action: cancel_order
# Action Input: {"order_id": "12345"}
# Observation: Order cancelled successfully
# Thought: I can now respond to the user
# Final Answer: Your order #12345 has been cancelled
```

**2. RAG for Knowledge Base**
```python
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# Store company docs in vector DB
vectorstore = Pinecone.from_documents(company_docs, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True  # Show where answer came from
)

# Agent can search company knowledge
@tool
def search_company_docs(query: str) -> str:
    """Search internal documentation"""
    result = qa_chain.invoke({"query": query})
    return result["result"]
```

**3. LangGraph for Complex Workflows**
```python
from langgraph.graph import StateGraph

# Multi-step process
workflow = StateGraph(SupportState)

workflow.add_node("classify", classify_ticket)
workflow.add_node("research", research_issue)
workflow.add_node("resolve", resolve_issue)
workflow.add_node("followup", send_followup)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "research")
workflow.add_conditional_edges(
    "research",
    should_resolve_now,
    {True: "resolve", False: "escalate"}
)
workflow.add_edge("resolve", "followup")

app = workflow.compile()
```

**4. Dynamic Prompts Based on Store**
```python
from langchain_core.prompts import ChatPromptTemplate

def create_store_agent(store_config: dict):
    """Create agent customized for each store"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a customer service agent for {company_name}.
        
        Company values: {values}
        Return policy: {return_policy}
        Tone: {tone}
        
        Special instructions:
        {special_instructions}
        """),
        ("human", "{input}"),
    ])
    
    # Each store gets custom agent
    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt.partial(**store_config)
    )
    
    return agent
```

**5. Evaluation Chain**
```python
from langchain.chains import LLMChain

eval_chain = LLMChain(
    llm=ChatAnthropic(model="claude-opus-4"),  # Stronger model for evaluation
    prompt=evaluation_prompt
)

def evaluate_agent_response(ticket_data: dict):
    """Evaluate agent performance"""
    
    evaluation = eval_chain.invoke({
        "conversation": ticket_data["conversation"],
        "resolution": ticket_data["resolution"],
        "customer_feedback": ticket_data.get("feedback")
    })
    
    # Store for training
    if evaluation["quality_score"] < 7:
        add_to_improvement_queue(ticket_data, evaluation)
    
    return evaluation
```

### 5.3 The "AI Manager" System

This is probably their most innovative piece:

```python
class AIManagerSystem:
    """AI that improves other AIs"""
    
    def __init__(self):
        self.supervisor_llm = ChatAnthropic(model="claude-opus-4")
        self.agent_performance_db = PerformanceDatabase()
    
    def analyze_agent_performance(self, agent_id: str, time_period: str):
        """Analyze how an agent is performing"""
        
        metrics = self.agent_performance_db.get_metrics(agent_id, time_period)
        
        analysis_prompt = f"""
        Analyze this agent's performance:
        
        Metrics:
        - Resolution rate: {metrics['resolution_rate']}
        - Avg response time: {metrics['avg_response_time']}
        - Customer satisfaction: {metrics['csat']}
        - Escalation rate: {metrics['escalation_rate']}
        
        Recent interactions:
        {metrics['recent_conversations']}
        
        Identify:
        1. What is the agent doing well?
        2. What needs improvement?
        3. Specific examples of good/bad responses
        4. Recommended changes to prompts/tools
        """
        
        analysis = self.supervisor_llm.invoke(analysis_prompt)
        return analysis
    
    def generate_improved_prompt(self, agent_id: str, analysis: dict):
        """Create better prompt based on analysis"""
        
        current_prompt = get_agent_prompt(agent_id)
        
        improvement_prompt = f"""
        Current agent prompt:
        {current_prompt}
        
        Performance analysis:
        {analysis}
        
        Generate an improved version of the prompt that:
        - Addresses identified weaknesses
        - Maintains strengths
        - Adds specific guidance based on problem examples
        
        Return only the new prompt.
        """
        
        new_prompt = self.supervisor_llm.invoke(improvement_prompt)
        
        return new_prompt
    
    def a_b_test_prompts(self, agent_id: str, old_prompt: str, new_prompt: str):
        """Test which prompt performs better"""
        
        # Create two agent versions
        agent_a = create_agent_with_prompt(old_prompt)
        agent_b = create_agent_with_prompt(new_prompt)
        
        # Route 50% of traffic to each
        # Track performance over 1 week
        # Automatically pick winner
        
        pass  # Implementation omitted for brevity
    
    def auto_improve_loop(self):
        """Continuously improve agents"""
        
        while True:
            # Every week
            time.sleep(604800)  # 1 week
            
            for agent_id in get_all_agents():
                # Analyze performance
                analysis = self.analyze_agent_performance(agent_id, "1_week")
                
                # If underperforming, generate improvements
                if analysis['needs_improvement']:
                    new_prompt = self.generate_improved_prompt(agent_id, analysis)
                    
                    # A/B test
                    self.a_b_test_prompts(agent_id, old_prompt, new_prompt)
```

### 5.4 Self-Serve Training Interface

How non-technical users train agents:

```python
class SelfServeTrainer:
    """Let users train agents without writing code"""
    
    def correct_agent_response(
        self,
        ticket_id: str,
        agent_response: str,
        correct_response: str,
        user_id: str
    ):
        """User corrects agent's response"""
        
        # Store correction
        correction = {
            "ticket_id": ticket_id,
            "agent_said": agent_response,
            "should_say": correct_response,
            "context": get_ticket_context(ticket_id),
            "timestamp": datetime.now()
        }
        
        save_correction(user_id, correction)
        
        # After N corrections, auto-improve prompt
        corrections = get_recent_corrections(user_id, limit=10)
        
        if len(corrections) >= 10:
            self.update_agent_from_corrections(user_id, corrections)
    
    def update_agent_from_corrections(self, user_id: str, corrections: list):
        """Improve agent based on user corrections"""
        
        current_prompt = get_user_agent_prompt(user_id)
        
        improvement_prompt = f"""
        Current prompt:
        {current_prompt}
        
        The user has corrected the agent {len(corrections)} times:
        
        {format_corrections(corrections)}
        
        Update the prompt to prevent these mistakes.
        Specifically add guidance about:
        - {extract_patterns(corrections)}
        
        Return the improved prompt.
        """
        
        new_prompt = llm.invoke(improvement_prompt)
        
        # Save new prompt
        update_user_agent_prompt(user_id, new_prompt)
        
        # Notify user
        notify_user(user_id, "Your agent has been automatically improved based on your corrections!")
    
    def learn_from_example(
        self,
        user_id: str,
        situation: str,
        example_response: str
    ):
        """User provides example of good response"""
        
        # Add to few-shot examples
        add_few_shot_example(user_id, {
            "input": situation,
            "output": example_response
        })
        
        # Agent will now use this as reference
```

---

## Part 6: Advanced Patterns

### 6.1 Multi-Agent Systems

**Problem:** One agent can't do everything well

**Solution:** Specialized agents that work together

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class SupportState(TypedDict):
    ticket_id: str
    customer_message: str
    classification: str
    resolution: str

# Agent 1: Triage
triage_agent = create_tool_calling_agent(
    llm,
    [],  # No tools, just classification
    triage_prompt
)

# Agent 2: Refund specialist
refund_agent = create_tool_calling_agent(
    llm,
    [process_refund, check_eligibility],
    refund_prompt
)

# Agent 3: Technical support
tech_agent = create_tool_calling_agent(
    llm,
    [check_logs, run_diagnostics],
    tech_prompt
)

# Agent 4: Escalation
escalation_agent = create_tool_calling_agent(
    llm,
    [create_human_ticket, notify_manager],
    escalation_prompt
)

# Orchestrator
def route_to_specialist(state: SupportState):
    """Route to appropriate specialist agent"""
    
    classification = state["classification"]
    
    if classification == "refund":
        return "refund_agent"
    elif classification == "technical":
        return "tech_agent"
    elif classification == "complex":
        return "escalation_agent"
    else:
        return "general_agent"

# Build workflow
workflow = StateGraph(SupportState)

workflow.add_node("triage", triage_agent)
workflow.add_node("refund_agent", refund_agent)
workflow.add_node("tech_agent", tech_agent)
workflow.add_node("escalation_agent", escalation_agent)

workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    route_to_specialist
)

app = workflow.compile()
```

### 6.2 Human-in-the-Loop

```python
from langgraph.checkpoint import MemorySaver

# Add checkpointing for human review
memory = MemorySaver()

workflow = StateGraph(SupportState, checkpoint=memory)

# Add approval node
def requires_human_approval(state: SupportState) -> bool:
    """Check if action needs human approval"""
    
    # High-value refunds need approval
    if state.get("refund_amount", 0) > 500:
        return True
    
    # Angry customers need approval
    if state.get("sentiment") == "very_negative":
        return True
    
    return False

def wait_for_approval(state: SupportState):
    """Pause execution for human approval"""
    
    # This actually pauses the workflow!
    # Human reviews in UI and approves/rejects
    
    return state

workflow.add_node("needs_approval", wait_for_approval)

workflow.add_conditional_edges(
    "process_refund",
    requires_human_approval,
    {
        True: "needs_approval",
        False: "send_confirmation"
    }
)

app = workflow.compile(checkpointer=memory)

# Agent processes ticket
result = app.invoke({"ticket_id": "123", ...})

# If it needed approval, execution paused
# Human reviews and approves
app.invoke({...}, config={"thread_id": "123"})  # Resumes from checkpoint
```

### 6.3 RAG with Re-Ranking

**Problem:** Simple vector search isn't always accurate

**Solution:** Re-rank results with LLM

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Compressor/re-ranker
compressor = LLMChainExtractor.from_llm(llm)

# Combined retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use it
docs = compression_retriever.get_relevant_documents(
    "What's the return policy for electronics?"
)

# Flow:
# 1. Vector search gets 10 potentially relevant docs
# 2. LLM re-ranks and compresses them
# 3. Returns only the most relevant 2-3 docs
```

### 6.4 Fallback Chains

```python
from langchain_core.runnables import RunnableWithFallbacks

# Primary chain (best but sometimes unavailable)
primary_chain = create_tool_calling_agent(
    ChatAnthropic(model="claude-opus-4"),
    tools,
    prompt
)

# Fallback chain (cheaper, always available)
fallback_chain = create_tool_calling_agent(
    ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools,
    prompt
)

# Create chain with fallback
chain = primary_chain.with_fallbacks([fallback_chain])

# If primary fails/unavailable, automatically uses fallback
result = chain.invoke({"input": "Help me"})
```

### 6.5 Semantic Caching

**Problem:** Regular caching only works for exact matches

**Solution:** Cache by meaning

```python
from langchain_community.cache import GPTCache
import gptcache

# Initialize semantic cache
gptcache.init(
    pre_embedding_func=gptcache.embedding.openai.openai_embedding,
    similarity_evaluation=gptcache.similarity_evaluation.simple_similarity,
    data_manager=gptcache.manager.get_data_manager()
)

set_llm_cache(GPTCache())

# These are semantically similar, so second uses cache:
llm.invoke("What's your return policy?")
llm.invoke("How do I return items?")  # Cache hit!
llm.invoke("Can I return products?")  # Cache hit!
```

---

## Part 7: Complete Project Examples

### 7.1 Project: Full Customer Support Agent

```python
# Complete, production-ready customer support agent

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import PostgresChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
import os

# ========== TOOLS ==========

@tool
def get_order_status(order_id: str) -> str: