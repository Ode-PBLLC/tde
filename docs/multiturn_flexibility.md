# Multi-Turn Conversation and LLM Flexibility Analysis

## Current Architecture Assessment

### Multi-Turn Conversation Support: ❌ Poor

**Stateless Design**: Each API request (`/query`, `/query/stream`) is completely independent with no conversation memory or context persistence.

**No Session Management**: The system has no concept of user sessions, conversation IDs, or persistent state between requests.

**Internal Tool Context Only**: The `messages` array in `mcp_chat.py:901-1400` is only used for internal tool calling within a single query, not for cross-query conversation history.

### LLM Abstraction and Routing: ❌ Limited

**Hardcoded Provider**: Direct instantiation of `anthropic.Anthropic()` in `MultiServerClient` class (`mcp_chat.py:905`).

**Fixed Model Selection**: Hardcoded calls to `claude-3-5-haiku-latest` throughout the codebase (`mcp_chat.py:1080, 1296, 1702, 2029`).

**No Provider Interface**: No abstraction layer for swapping LLM providers during peak usage or for load balancing.

## Proposed Solution: Session-Based Context with LLM Routing

### Core Concept

**Previous Response as Context**: Store the last response from each session and include it as context for subsequent queries, enabling natural multi-turn conversations without API changes.

**LLM Provider Abstraction**: Create a flexible LLM interface that can route requests to different providers (Anthropic, OpenAI, etc.) during peak usage.

### Key Constraints

- ✅ **API Compatibility**: Cannot change existing `/query` and `/query/stream` endpoints
- ✅ **No User Management**: Conversations only need to last within browser sessions
- ✅ **Frontend Transparency**: Changes must be invisible to the existing frontend

## Implementation Plan

### 1. Session-Based Context Management

#### Session Tracking Strategy
```python
# Add to api_server.py
from fastapi import Request
import hashlib

def get_session_id(request: Request) -> str:
    """Generate session ID from IP address and User-Agent for basic session tracking."""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    session_data = f"{client_ip}:{user_agent}"
    return hashlib.md5(session_data.encode()).hexdigest()[:16]
```

#### Context Storage Structure
```python
# In-memory session storage (add to api_server.py)
session_contexts = {}  # {session_id: {"history": [...], "last_updated": datetime}}

class SessionContext:
    def __init__(self):
        self.history = []  # List of {query, response, timestamp}
        self.last_updated = datetime.now()
        
    def add_interaction(self, query: str, response: dict):
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 2 interactions to manage token limits
        if len(self.history) > 2:
            self.history = self.history[-2:]
        self.last_updated = datetime.now()
```

#### Context Injection in Query Processing
```python
# Modify process_query in api_server.py
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    session_id = get_session_id(http_request)
    
    # Get previous context if available
    previous_context = None
    if session_id in session_contexts:
        context = session_contexts[session_id]
        if context.history:
            previous_context = context.history[-1]  # Last interaction
    
    # Process query with context
    full_result = await run_query(request.query, previous_context=previous_context)
    
    # Store this interaction
    if session_id not in session_contexts:
        session_contexts[session_id] = SessionContext()
    session_contexts[session_id].add_interaction(request.query, full_result)
```

### 2. LLM Provider Abstraction Layer

#### Abstract LLM Interface
```python
# Add to mcp/llm_providers.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    async def create_message(self, model: str, max_tokens: int, system: str, messages: List[Dict]) -> Any:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    
    async def create_message(self, model: str, max_tokens: int, system: str, messages: List[Dict]) -> Any:
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages
        )
    
    def get_available_models(self) -> List[str]:
        return ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def create_message(self, model: str, max_tokens: int, system: str, messages: List[Dict]) -> Any:
        # Convert Anthropic format to OpenAI format
        openai_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=openai_messages
        )
        return response
    
    def get_available_models(self) -> List[str]:
        return ["gpt-4o-mini", "gpt-4o"]
```

#### LLM Router with Load Balancing
```python
# Add to mcp/llm_router.py
import random
from typing import Optional
from .llm_providers import LLMProvider, AnthropicProvider, OpenAIProvider

class LLMRouter:
    def __init__(self):
        self.providers = {}
        self.provider_weights = {}
        self.setup_providers()
    
    def setup_providers(self):
        """Initialize available providers based on environment variables."""
        import os
        
        # Primary provider (Anthropic)
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers["anthropic"] = AnthropicProvider()
            self.provider_weights["anthropic"] = 1.0
        
        # Fallback provider (OpenAI)
        if os.getenv("OPENAI_API_KEY"):
            self.providers["openai"] = OpenAIProvider()
            self.provider_weights["openai"] = 0.8
    
    async def route_request(self, model: str, max_tokens: int, system: str, messages: List[Dict], preferred_provider: str = "anthropic"):
        """Route request to available provider with fallback logic."""
        
        # Try preferred provider first
        if preferred_provider in self.providers:
            try:
                return await self.providers[preferred_provider].create_message(model, max_tokens, system, messages)
            except Exception as e:
                print(f"Provider {preferred_provider} failed: {e}, trying fallback...")
        
        # Try other providers as fallback
        for provider_name, provider in self.providers.items():
            if provider_name != preferred_provider:
                try:
                    # Map model names between providers
                    mapped_model = self._map_model(model, provider_name)
                    return await provider.create_message(mapped_model, max_tokens, system, messages)
                except Exception as e:
                    print(f"Fallback provider {provider_name} failed: {e}")
                    continue
        
        raise Exception("All LLM providers failed")
    
    def _map_model(self, model: str, provider: str) -> str:
        """Map model names between providers."""
        model_mapping = {
            "anthropic": {
                "claude-3-5-haiku-latest": "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-latest": "claude-3-5-sonnet-latest"
            },
            "openai": {
                "claude-3-5-haiku-latest": "gpt-4o-mini",
                "claude-3-5-sonnet-latest": "gpt-4o"
            }
        }
        return model_mapping.get(provider, {}).get(model, "gpt-4o-mini")
```

### 3. Context-Aware Query Processing

#### Modify MultiServerClient for Context Support
```python
# Modify mcp_chat.py MultiServerClient class
class MultiServerClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        # Replace hardcoded anthropic client with router
        from .llm_router import LLMRouter
        self.llm_router = LLMRouter()
        self.citation_registry = CitationRegistry()
    
    async def process_query(self, query: str, previous_context: Optional[Dict] = None):
        """Process query with optional previous context."""
        
        # Build system prompt with context
        system_prompt = self._build_system_prompt_with_context(previous_context)
        
        # Initial messages
        messages = [{"role": "user", "content": query}]
        
        # Rest of the existing process_query logic...
        # Replace self.anthropic.messages.create calls with self.llm_router.route_request
        
    def _build_system_prompt_with_context(self, previous_context: Optional[Dict] = None) -> str:
        """Build system prompt including previous conversation context if available."""
        base_prompt = """You are a climate policy expert assistant..."""  # Existing prompt
        
        if previous_context:
            context_addition = f"""
            
Previous conversation context:
Query: {previous_context['query']}
Your previous response: {self._summarize_previous_response(previous_context['response'])}

You may reference this previous information if relevant to the current query, but only if it adds value. Do not force connections that don't exist naturally.
"""
            return base_prompt + context_addition
        
        return base_prompt
    
    def _summarize_previous_response(self, response: Dict) -> str:
        """Create a concise summary of the previous response for context."""
        # Extract key information from structured response
        modules = response.get("formatted_response", {}).get("modules", [])
        summary_parts = []
        
        for module in modules[:3]:  # Limit to first 3 modules
            if module.get("type") == "text":
                texts = module.get("texts", [])
                if texts:
                    summary_parts.append(texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0])
        
        return " | ".join(summary_parts) if summary_parts else "Previous query processed successfully."
```

### 4. Session Cleanup and Management

#### Automatic Session Cleanup
```python
# Add to api_server.py
import asyncio
from datetime import datetime, timedelta

async def cleanup_old_sessions():
    """Background task to clean up sessions older than 1 hour."""
    while True:
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, context in session_contexts.items():
            if current_time - context.last_updated > timedelta(hours=1):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del session_contexts[session_id]
        
        print(f"Cleaned up {len(expired_sessions)} expired sessions")
        await asyncio.sleep(300)  # Clean up every 5 minutes

@app.on_event("startup")
async def startup_event():
    """Enhanced startup to include session cleanup."""
    try:
        print("Warming up global MCP client...")
        await get_global_client()
        print("Global MCP client warmed up successfully")
        
        # Start session cleanup task
        asyncio.create_task(cleanup_old_sessions())
        print("Session cleanup task started")
    except Exception as e:
        print(f"Warning: Failed to warm up global MCP client: {e}")
```

## Implementation Benefits

### Multi-Turn Conversations
- ✅ **Natural Flow**: Users can ask follow-up questions that reference previous responses
- ✅ **Context Awareness**: LLM can build on previous information when relevant
- ✅ **API Compatible**: No changes required to frontend or existing endpoints
- ✅ **Memory Efficient**: Only stores last 1-2 interactions per session

### LLM Flexibility
- ✅ **Provider Agnostic**: Can route between Anthropic, OpenAI, and other providers
- ✅ **Automatic Fallback**: Graceful degradation when primary provider is unavailable
- ✅ **Load Distribution**: Can distribute load across multiple providers during peak usage
- ✅ **Cost Optimization**: Can route to cheaper providers for simpler queries

### Operational Benefits
- ✅ **High Availability**: Multiple provider fallback prevents service disruptions
- ✅ **Scalability**: Can handle peak loads by utilizing multiple LLM providers
- ✅ **Monitoring Ready**: Provider switching can be logged and monitored
- ✅ **Configuration Driven**: Provider preferences can be adjusted via environment variables

## File Modifications Required

1. **`api_server.py`**: Add session tracking, context storage, and modify query endpoints
2. **`mcp/mcp_chat.py`**: Update MultiServerClient to use LLM router and handle context
3. **`mcp/llm_providers.py`**: New file for provider abstraction layer
4. **`mcp/llm_router.py`**: New file for routing and load balancing logic

## Configuration

### Environment Variables
```bash
# Primary provider
ANTHROPIC_API_KEY=your_anthropic_key

# Fallback provider
OPENAI_API_KEY=your_openai_key

# Optional: Provider preferences
LLM_PRIMARY_PROVIDER=anthropic
LLM_ENABLE_FALLBACK=true
```

This implementation provides robust multi-turn conversation capabilities and LLM provider flexibility while maintaining complete compatibility with the existing frontend API.