import asyncio
import anthropic
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional
import time
import os
from dotenv import load_dotenv
import json
from textwrap import fill
import streamlit as st
import pandas as pd
import plotly.express as px

load_dotenv()

MAX_CTX_CHARS = 18_000          # hard cap – keep below Claude-Haiku context
WIDE = 88          # tweak for your terminal width
SUMMARY_PROMPT = (
    "You are a climate-policy expert. "
    "Assume the reader wants the big picture and key linkages.\n\n"
)

def _fmt_sources(sources):
    """
    Return a pretty string for `sources` which may be a mix of:
      • plain strings (legacy metadata)
      • dicts like {"doc_id": "D123", "passage_id": "P456"}
    """
    if not sources:
        return "— no sources captured —"

    rows = []
    for src in sources:
        if isinstance(src, dict):
            rows.append(f"{src.get('doc_id','?'):>8}  {src.get('passage_id','?'):>10}")
        else:                                # plain str or other
            rows.append(str(src))
    # header for ID pairs
    if any(isinstance(s, dict) for s in sources):
        rows.insert(0, f"{'DOC ID':>8}  {'PASSAGE':>10}")
        rows.insert(1, "-"*20)
    return "\n".join(rows)

def harvest_sources(payload):
    """
    Accepts result.content (could be list/dict/str) and
    returns a list of {doc_id, passage_id} records.
    """
    out = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "passage_id" in item:
                out.append({
                    "doc_id":     item.get("doc_id") or item.get("document_id"),
                    "passage_id": item["passage_id"],
                })
            # PathContext hop → hop["passages"] list[str] (no IDs) → skip
    elif isinstance(payload, dict):
        if "passage_id" in payload:
            out.append({
                "doc_id":     payload.get("doc_id") or payload.get("document_id"),
                "passage_id": payload["passage_id"],
            })
    return out


class CPR_Client:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic()
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # shuts down stdio_client, ClientSession, etc. in the *same* task
        await self.exit_stack.aclose()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        print("connected to server")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print("Initialized session")

    async def call_tool(self, tool_name: str, tool_args: dict):
        async with self.session:                         # was self.client
            return await self.session.call_tool(tool_name, tool_args)

        
    async def process_query(self, query: str):
        # --- Temporary hardcoding for chart visualization --- 
        DUMMY_DATASET_ID = "DUMMY_DATASET_EXTREME_WEATHER"
        if query.lower() == "show dummy chart":
            print(f"Hardcoded trigger: Querying for dataset ID: {DUMMY_DATASET_ID}")
            # Fall through to the main logic, but ensure the AI targets this ID if it uses GetDatasetContent
            # This specific hardcoding will now rely on the general GetDatasetContent parsing below.
            # For true hardcoding, we'd simulate the AI making the *correct* call.
            # To ensure it uses the GetDatasetContent, we can subtly change the query for the AI if needed,
            # or rely on the fact that the user *wants* this specific dataset for this query.
            # For now, we assume if query is "show dummy chart", the AI will be guided/already knows.
            # The main purpose of this block is now just for a print statement.
            pass # Let the normal flow attempt to get this via AI if possible, or adjust query for AI.

        messages = [
            {"role": "user", 
            "content": query} # Original query
        ]
        
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        system_prompt = """
            You are a climate policy expert.
            When a user asks for a dataset by a descriptive name, try to find its actual ID using graph navigation tools first (like GetConceptGraphNeighbors).
            Then use the GetDatasetContent tool with the discovered 'node_id'.
            If you are asked to "show dummy chart", the ID for GetDatasetContent is DUMMY_DATASET_EXTREME_WEATHER.
            """

        if query.lower() == "show dummy chart": # Override messages for dummy chart query
             messages = [
                {"role": "user", "content": "Get the content of dataset DUMMY_DATASET_EXTREME_WEATHER."}
            ]

        response = self.anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
            tools=available_tools
        )

        final_text = []
        sources_used = []
        context_chunks = []   # every tool_result.content goes in here
        passage_sources = []        # each element: {"doc_id": …, "passage_id": …}
        chart_data = None           # To store data for charting

        while True:
            assistant_message_content = []

            for content in response.content:
                if content.type == "text":
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Pretty print of the tool and its arguments
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    result = await self.session.call_tool(tool_name, tool_args)

                    # Generalized parsing for GetDatasetContent output
                    if tool_name == "GetDatasetContent":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, list):
                                        # Check if it's a list of dicts (actual data)
                                        if all(isinstance(item, dict) for item in parsed_content):
                                            chart_data = parsed_content # Assign to chart_data
                                            print(f"Successfully parsed chart data from {tool_name}: {len(chart_data)} records")
                                        else:
                                            print(f"Content from {tool_name} is a list, but not of dictionaries: {first_content_block.text[:100]}...")
                                    else:
                                        # This means the .text was valid JSON, but not a list (e.g. a string like "Dataset not found")
                                        print(f"Parsed JSON from {tool_name} is not a list: {first_content_block.text[:100]}...")
                                except json.JSONDecodeError:
                                    # This means .text was not valid JSON (e.g. plain string "Dataset not found")
                                    print(f"Content from {tool_name} is not valid JSON: {first_content_block.text[:100]}...")
                            else:
                                print(f"Content block from {tool_name} is not TextContent or lacks .text attribute.")
                        else:
                            print(f"{tool_name} did not return expected content structure: {result.content}")
                    
                    # final_text.append(f"[Calling tool {tool_name} with args {tool_args}]") # Removed for cleaner UI
                    try:
                        context_chunks.append(json.dumps(result.content, ensure_ascii=False))
                    except Exception:
                        pass

                    # 1) legacy metadata capture
                    if tool_name.lower() == "getmetadata":
                        sources_used.append(result.content)

                    # 2) NEW: collect passage/document IDs anywhere they appear
                    passage_sources.extend(harvest_sources(result.content))

                    # Attach tool_use to assistant message
                    assistant_message_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }]
                    })

                    # Break early and send updated messages to Claude
                    break
            else:
                # No tool_use found → conversation complete
                messages.append({"role": "assistant", "content": assistant_message_content})
                break

            # Ask Claude for the next step
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools,
            )

        # --- final synthesis -------------------------------------------------
        if context_chunks:
            # Trim if context explodes
            joined_ctx = "\n\n".join(context_chunks)
            if len(joined_ctx) > MAX_CTX_CHARS:
                joined_ctx = joined_ctx[:MAX_CTX_CHARS] + "\n\n[truncated]"
            
            summary_resp = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                system="You are an expert climate-policy summariser.",
                messages=[
                    {"role": "user", "content": SUMMARY_PROMPT + joined_ctx}
                ],
            )
            summary_text = summary_resp.content[0].text.strip()
            final_text.append("\n\n## Summary\n" + summary_text)

        # de-dupe sources
        uniq_passages = {(p["doc_id"], p["passage_id"]): p for p in passage_sources}
        if uniq_passages:
            sources_used.extend(uniq_passages.values())


        return {
            "response": "\n".join(final_text),
            "sources": sources_used or ["No source captured"],
            "chart_data": chart_data  # Add chart_data to the return dict
        }
        
async def run_query(q: str):
    async with CPR_Client() as client:          # ← guarantees cleanup
        await client.connect_to_server("cpr_kg_server.py")
        return await client.process_query(q)

def pretty_print(result: dict):
    """
    Nicely prints the 'response' markdown plus a Sources block.
    """
    separator = "\n" + "="*WIDE + "\n"
    print(separator)
    print(fill(result["response"], width=WIDE, replace_whitespace=False))
    print(separator)
    print("SOURCES")
    print(_fmt_sources(result["sources"]))
    print(separator)

async def main_streamlit():
    st.title("Climate Policy Radar Chat")

    query = st.text_input("Enter your query:")

    if st.button("Run Query"):
        if query:
            with st.spinner("Processing your query..."):
                result = await run_query(query)
            
            st.markdown("## Response")
            st.markdown(result["response"], unsafe_allow_html=True)
            
            # Display chart if data is available
            chart_data_from_result = result.get("chart_data")
            
            # --- Debug prints for chart_data ---
            print("--- Debug main_streamlit ---")
            print(f"chart_data_from_result: {chart_data_from_result}")
            print(f"type(chart_data_from_result): {type(chart_data_from_result)}")
            if isinstance(chart_data_from_result, list) and chart_data_from_result:
                print(f"type(chart_data_from_result[0]): {type(chart_data_from_result[0])}")
            print("---------------------------")
            # --- End Debug prints ---
            
            # Be more specific: check if it's a list and not empty, and ideally a list of dicts
            if isinstance(chart_data_from_result, list) and chart_data_from_result and isinstance(chart_data_from_result[0], dict):
                st.markdown("## Interactive Chart")
                try:
                    df = pd.DataFrame(result["chart_data"])
                    if not df.empty and 'impact_rating' in df.columns and 'type' in df.columns:
                        # Example: Bar chart of average impact rating by event type
                        # You can customize this based on the actual data structure
                        if 'event_id' not in df.columns:
                             df['event_id'] = df.index # or some other unique identifier

                        fig = px.bar(df, x="type", y="impact_rating", 
                                     color="type", 
                                     title="Extreme Weather Event Impact Ratings",
                                     labels={"type":"Event Type", "impact_rating":"Impact Rating"},
                                     hover_data=["event_id", "description", "location", "year"])
                        st.plotly_chart(fig, use_container_width=True)
                    elif not df.empty:
                        st.write("Data available but not in expected format for the default chart.")
                        st.dataframe(df) # Display raw data as a fallback
                    else:
                        st.write("No data to display in chart.")
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
                    st.write("Raw chart data:")
                    st.json(result["chart_data"]) # show the raw data if charting fails

            st.markdown("## Sources")
            st.text_area("Sources", _fmt_sources(result["sources"]), height=200)
        else:
            st.warning("Please enter a query.")

def main(): # Renaming original main
    # This function will now only be called if not running in streamlit context
    # or for testing purposes, so we can keep the original query
    # If you want to remove it, you can do so.
    async def run_original_main():
        result = await run_query(
            "How does `extreme weather` relate to `people with limited assets`? Are there passages where both are mentioned?"
        )
        pretty_print(result)
    asyncio.run(run_original_main())

if __name__ == "__main__":
    # Ensure asyncio event loop is properly managed for Streamlit
    # Check if running in Streamlit context, if not, run original main
    try:
        import streamlit.runtime.scriptrunner as scr
        if not scr.get_script_run_ctx():
            main() # call original main
        else:
            asyncio.run(main_streamlit()) # run streamlit main
    except ModuleNotFoundError:
        # Fallback if streamlit is not installed or scriptrunner path changes
        main() # call original main
