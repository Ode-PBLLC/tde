# Multi-Turn Conversation API Documentation

## Overview
The API now supports multi-turn conversations through session management. This allows users to have contextual conversations where the system remembers previous interactions.

## Key Parameter: `conversation_id`

The new parameter your frontend needs to handle is **`conversation_id`** (optional string).

## API Changes

### 1. Query Endpoint (`POST /query`)

**Request Body:**
```json
{
  "query": "Your question here",
  "include_thinking": false,
  "conversation_id": "optional-session-id"  // NEW PARAMETER
}
```

**Parameters:**
- `conversation_id` (optional): String identifier for the conversation session
  - If omitted, a new session is created
  - If provided, continues existing conversation
  - Server returns the active session ID in response

### 2. Streaming Endpoint (`POST /query/stream`)

**Request Body:**
```json
{
  "query": "Your question here",
  "conversation_id": "optional-session-id"  // NEW PARAMETER
}
```

**Streaming Response:**
The first event in the stream (if `conversation_id` is new or different) will be:
```json
{
  "type": "conversation_id", 
  "data": {
    "conversation_id": "actual-conversation-id-from-server"
  }
}
```

## Session Management

### Session Behavior
- **Session TTL**: 20 minutes of inactivity (then auto-expires)
- **Context Window**: Last 2 conversation turns (4 messages: 2 user + 2 assistant)
- **Auto-creation**: If no `conversation_id` provided, server creates new session
- **Auto-cleanup**: Expired sessions are cleaned up automatically

### Frontend Implementation Guidelines

1. **Initial Request** (New Conversation):
   ```javascript
   const response = await fetch('/query/stream', {
     method: 'POST',
     body: JSON.stringify({
       query: "What is climate change?"
       // No conversation_id - server will create one
     })
   });
   
   // Parse streaming response to get conversation_id from first event
   ```

2. **Follow-up Request** (Continue Conversation):
   ```javascript
   const response = await fetch('/query/stream', {
     method: 'POST',
     body: JSON.stringify({
       query: "Tell me more about its impacts",
       conversation_id: savedConversationId  // Use the conversation_id from previous response
     })
   });
   ```

3. **Start New Conversation**:
   ```javascript
   // Simply omit the conversation_id to start fresh
   const response = await fetch('/query/stream', {
     method: 'POST',
     body: JSON.stringify({
       query: "Let's talk about renewable energy"
       // No conversation_id - gets new session
     })
   });
   ```

## Implementation Tips

1. **Store Conversation ID**: Save the `conversation_id` returned by the server in your frontend state
2. **Handle Session Expiry**: If a session expires (20 min inactivity), the server will create a new one automatically
3. **New Chat Button**: Simply omit `conversation_id` in the next request to start a fresh conversation
4. **Error Handling**: If server doesn't recognize a `conversation_id`, it creates a new session

## Example Frontend Flow

```javascript
class ChatClient {
  constructor() {
    this.currentConversationId = null;
  }
  
  async sendMessage(query, startNewConversation = false) {
    const requestBody = {
      query: query
    };
    
    // Include conversation_id only if we have one and not starting new
    if (this.currentConversationId && !startNewConversation) {
      requestBody.conversation_id = this.currentConversationId;
    }
    
    const response = await fetch('/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });
    
    // Handle streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          // Check for conversation_id event
          if (data.type === 'conversation_id') {
            this.currentConversationId = data.data.conversation_id;
            console.log('New Conversation ID:', this.currentConversationId);
          } else {
            // Handle other event types (text, citations, etc.)
            this.handleEvent(data);
          }
        }
      }
    }
  }
  
  // Start a new conversation
  startNewChat() {
    this.currentConversationId = null;
  }
}
```

## What Happens Behind the Scenes

When you include a `conversation_id`:
1. Server retrieves the last 2 turns of conversation (if they exist)
2. Includes this context when processing the query
3. The AI can reference previous questions and answers
4. Provides more coherent, contextual responses

## Backend Logging

The server logs all conversations to `conversation_logs.csv` for analytics, including:
- Timestamp
- Conversation ID  
- Turn number
- Query and response summaries
- Session duration
- Token usage

## Migration Notes

- **Backward Compatible**: The `conversation_id` parameter is optional
- **No Breaking Changes**: Existing integrations will continue to work
- **Gradual Adoption**: You can add conversation support incrementally

## Summary for Frontend Team

**What you need to do:**
1. Capture the `conversation_id` from the first streaming event (type: "conversation_id")
2. Include it as `conversation_id` in subsequent requests to maintain context
3. Omit `conversation_id` when user wants to start a new conversation

**The parameter name to use:** `conversation_id`

That's it! The server handles all the session management, expiry, and context windowing automatically.