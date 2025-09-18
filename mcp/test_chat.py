import anthropic
import asyncio
import sys
import os
from typing import Optional

from contextlib import AsyncExitStack
from dotenv import load_dotenv
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

class SimplifiedClient:
    """A client to connect to a single server and process queries."""

    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic(api_key=api_key)

    async def connect_and_run(self, query: str):
        """Connects to the server, processes the query, and closes the connection."""
        is_python = self.server_script_path.endswith('.py')
        if not is_python:
            raise ValueError("Server script must be a .py file")

        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_script_path],
            env=None
        )

        print(f"Connecting to server at {self.server_script_path}...")
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            print("Connection successful! Processing query...")
            
            result = await self.process_query(query)
            print("\n--- Response from Server ---")
            print(result["response"])
            print("\n--- Sources ---")
            print("\n".join(result["sources"]))

        finally:
            print("Closing connection.")
            await self.exit_stack.aclose()

    async def process_query(self, query: str):
        """Sends a query to the server and handles the response with tool use."""
        messages = [{"role": "user", "content": query}]
        
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        system_prompt = """You are a data-savvy assistant for decision makers in humanitarian and security contexts. You provide helpful national and historical context.

                You understand that security has many dimensions, and that the drivers of security: food, water, health, economy, climate, nature, conflict, politics, and inequality, are interrelated. 
                Even if the topic of interest is only relating to one topic, you should also obtain some contextual data on interrelated topics to provide in your final response.

                You have access to specialized tools to analyze a structured dataset of national indicators for Yemen (e.g. Poverty headcount ratio, life expectancy).
                You should always use the server's tool to get metadata about a given dataset, so you can understand its sourcing and limitations.

                You should **always** use the available tools to gather structured insights.
                When answering a complex question:
                - Start by getting the available indicators, areas, and time ranges as needed
                - Choose relevant indicators and identifying the time range and areas with available data
                - Then explore key trends, correlations, etc. using tools like 'GetTrendSummary' or 'CompareIndicators' 
                - Chain multiple tools if needed to reach your conclusion

                Be sure to highlight any areas where the data is limited or out of date, and/or where you are uncertain.
        
                Visualization: Use visualization tools (GetSolarCapacityVisualizationData, GetGistVisualizationData, GetLSEVisualizationData) for chart requests. System will generate charts and maps that the user can review, but that you will not see.

                Output: Synthesize tool results into comprehensive response. Don't narrate tool calling process. Include specific numbers and context.
        """

        while True:
            llm_response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest", # Using a more recent model
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
                tools=available_tools
            )

            final_text_parts = []
            sources_used = []
            tool_call_found = False

            for content in llm_response.content:
                if content.type == "text":
                    final_text_parts.append(content.text)
                elif content.type == "tool_use":
                    tool_call_found = True
                    tool_name = content.name
                    tool_args = content.input
                    print(f"-> Calling tool: {tool_name} with args: {tool_args}")

                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        messages.append({"role": "assistant", "content": llm_response.content})
                        print(result)

                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }]
                        })

                        if tool_name.lower() == "getmetadata":
                            try:
                                sources_used.append(result.content[0].text)
                            except (IndexError, AttributeError):
                                pass

                    except Exception as e:
                        print(f"Error calling tool {tool_name}: {e}")
                        messages.append({"role": "assistant", "content": llm_response.content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": f"Error: {e}"
                            }]
                        })
                    
                    break # Exit the for loop to get a new response with the tool result

            if not tool_call_found:
                # No more tool calls, conversation is complete
                return {
                    "response": "".join(final_text_parts),
                    "sources": sources_used or ["No source captured"]
                }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_chat.py \"what is tomorrow's precipitation forecast\"")
        sys.exit(1)

    query = sys.argv[1]
    print(query)
    
    # Specify the server script you want to connect to.
    # Change this to 'mics_server.py', 'wb_server.py', etc.
    server_to_use = "mcp/wmo_cli_server.py" 
    #server_to_use = os.environ.get("SERVER_TO_USE", "mcp/world_bank_server.py")

    client = SimplifiedClient(server_to_use)
    asyncio.run(client.connect_and_run(query))