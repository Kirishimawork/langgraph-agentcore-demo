import os
import re
import logging
from dotenv import load_dotenv

# LangGraph & LangChain imports
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# AWS AgentCore Memory imports
from langgraph_checkpoint_aws import (
    AgentCoreMemorySaver, 
    AgentCoreMemoryStore
)
# Tool decorator
from langchain.tools import tool

# Utility imports
from bedrock_utils import query_llm
from redshift_utils import (
    execute_query_redshift,
    execute_query_with_pagination,
    redshift_querys
)

# -------------------- Setup --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
MEMORY_ID = "tan_demo_memory-PVGrSIHDTG"
MODEL_ID = os.getenv('MODEL_ID', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')

CONNECTION_PARAM = {"WorkgroupName": os.getenv("WORKGROUP_NAME", "animal-food-data")}
DATABASE = os.getenv("DATABASE", "dev")
DB_USER = os.getenv("DB_USER", "tanpat")
SQL_MODEL_ID = "amazon.nova-pro-v1:0"

# -------------------- Core Tools (5 tools) --------------------

@tool
def get_database_schema(schema_name: str) -> str:
    """
    Retrieve the structure of database tables including columns and data types.
    
    Args:
        schema_name: The schema name to query (e.g., 'public', 'sales')
    
    Returns:
        Table structure information in CSV format
    """
    logger.info(f"*** get_database_schema: {schema_name} ***")
    try:
        sql = (
            "SELECT table_catalog,table_schema,table_name,column_name,"
            "ordinal_position,is_nullable,data_type "
            f"FROM information_schema.columns WHERE table_schema='{schema_name}'"
        )
        resp = execute_query_redshift(sql, CONNECTION_PARAM, DATABASE)
        result = redshift_querys(sql, resp, {}, CONNECTION_PARAM, DATABASE)
        return result[0]
    except Exception as e:
        logger.error(f"get_database_schema error: {e}")
        return f"Error: {e}"

@tool
def get_sample_data(schema_name: str, table_names: str) -> str:
    """
    Fetch sample data (3 rows) from specified tables to understand data structure.
    
    Args:
        schema_name: The schema containing the tables
        table_names: Comma-separated table names (e.g., "products,sales")
    
    Returns:
        Sample data from each table in CSV format
    """
    logger.info(f"*** get_sample_data: {table_names} ***")
    try:
        tables = [t.strip() for t in table_names.split(',')]
        sqls = [f"SELECT * FROM {schema_name}.{t} LIMIT 3" for t in tables]
        results = execute_query_with_pagination(sqls, CONNECTION_PARAM, DATABASE)
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"get_sample_data error: {e}")
        return f"Error: {e}"

@tool
def generate_sql_with_context(question: str, schema_info: str = "", sample_data: str = "") -> str:
    """
    Generate a SQL query based on user question, schema structure, and sample data.
    
    Args:
        question: The user's question in natural language
        schema_info: Database schema information
        sample_data: Sample data from relevant tables
    
    Returns:
        A valid PostgreSQL/Redshift SQL query
    """
    logger.info("*** generate_sql_with_context ***")
    try:
        prompts = f"""<s><<SYS>>[INST]
You are an expert PostgreSQL/Redshift developer working with a pet food sales database.

Database Context:
- Tables: Product_Catalog (pet food products), Sales_Transaction_Details (sales records)
- Product_Catalog columns: Product_ID, Product_Name, Pet_Type, Flavor, Size_kg, Price_THB
- Sales_Transaction columns: Transaction_ID, Date, Time, Product_ID, Quantity, Unit_Price, Discount_Rate, Total_Paid, Payment_Method, Store_Location, Customer_Tier

Schema Information:
########
{schema_info}
########

Sample Data:
########
{sample_data}
########
<</SYS>>

Instructions:
1. Always include schema name for tables (e.g., public.product_catalog)
2. Use only required columns for efficiency
3. Wrap column/table names with double quotes if they contain special characters
4. Don't reference 'dev' database explicitly
5. Use appropriate JOINs when querying multiple tables
6. Consider aggregations (SUM, COUNT, AVG) for analytical questions

Return ONLY the SQL query inside <sql> tags:
<sql>
your SQL query here
</sql>

User Question: {question}[/INST]"""

        response = query_llm(prompts, SQL_MODEL_ID)
        match = re.search(r"<sql>(.*?)(?:</sql>|$)", response, re.DOTALL)
        return match.group(1).strip().replace("\\", "") if match else response
    except Exception as e:
        logger.error(f"generate_sql_with_context error: {e}")
        return f"Error: {e}"

@tool
def quick_test_sql(sql_query: str) -> str:
    """
    Test if a SQL query is valid without retrieving full results.
    
    Args:
        sql_query: The SQL query to validate
    
    Returns:
        Success message or error details
    """
    logger.info("*** quick_test_sql ***")
    try:
        # Add LIMIT to make it quick
        test_query = sql_query.strip()
        if "LIMIT" not in test_query.upper():
            test_query += " LIMIT 1"
        
        resp = execute_query_redshift(test_query, CONNECTION_PARAM, DATABASE)
        return "SQL is valid ✓"
    except Exception as e:
        logger.error(f"quick_test_sql error: {e}")
        return f"SQL Failed: {e}"

@tool
def query_existing_table(sql_query: str) -> str:
    """
    Execute a SQL query and retrieve actual data from the database.
    
    Args:
        sql_query: The SQL query to execute
    
    Returns:
        Query results in CSV format
    """
    logger.info("*** query_existing_table: execute ***")
    try:
        resp = execute_query_redshift(sql_query, CONNECTION_PARAM, DATABASE)
        result = redshift_querys(sql_query, resp, {}, CONNECTION_PARAM, DATABASE)
        return result[0]
    except Exception as e:
        logger.error(f"query_existing_table error: {e}")
        return f"Error: {e}"

from langgraph.graph import MessagesState
from typing_extensions import NotRequired

class CustomState(MessagesState):
    schema_info: NotRequired[str]
    sample_data: NotRequired[str]

# -------------------- LangGraph Nodes --------------------

def should_continue(state: MessagesState):
    """
    Determine whether to continue calling tools or end the workflow.
    
    Args:
        state: Current agent state
    
    Returns:
        "continue" if tools need to be called, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # ✅ ใช้ getattr เพื่อหลีกเลี่ยง warning
    tool_calls = getattr(last_message, 'tool_calls', None)
    
    if not tool_calls:
        return "end"
    return "continue"

from langchain_core.messages import SystemMessage

def call_model(state: CustomState):
    cached_schema = state.get('schema_info', '')
    cached_sample = state.get('sample_data', '')
    
    logger.info(f"[call_model] Schema cached: {len(cached_schema)} chars | Sample cached: {len(cached_sample)} chars")
    
    system_msg = SystemMessage(content=f"""You are a Text-to-SQL expert assistant.

Cached Schema: {cached_schema[:200] if cached_schema else 'Not loaded'}
Cached Sample Data: {cached_sample[:200] if cached_sample else 'Not loaded'}

STRICT WORKFLOW (follow in order):
1. Load database schema/data: Call get_database_schema ONLY if NOT cached. Call get_sample_data ONLY if NOT cached.
2. Generate SQL once: Call generate_sql_with_context EXACTLY ONCE. Do NOT regenerate SQL.
3. Validate SQL: Call quick_test_sql to verify syntax. If it passes, STOP regenerating.
4. Execute: Call query_existing_table to get final results.

CRITICAL RULES:
- NEVER call generate_sql_with_context multiple times for the same query
- Once quick_test_sql passes, proceed to query_existing_table immediately
- If SQL has issues, explain to user instead of regenerating infinitely
- Reuse cached schema/sample data from previous questions""")
    
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    tool_calls = getattr(response, 'tool_calls', None)
    if tool_calls:
        tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in tool_calls]
        logger.info(f"[call_model] Tools called: {tool_names}")
    
    return {"messages": [response]}


def tools_with_state_update(state: CustomState):
    """Execute tools และอัปเดต state"""
    # เรียก tools ปกติ
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)

    # ดึงผลลัพธ์จาก tool messages
    # Preserve existing schema/sample values unless a tool explicitly updates them
    updates = {
        "messages": result["messages"],
        "schema_info": state.get("schema_info", ""),
        "sample_data": state.get("sample_data", ""),
    }

    for msg in result["messages"]:
        if hasattr(msg, 'name'):
            if msg.name == "get_database_schema":
                updates["schema_info"] = msg.content
                logger.info(f"[tools] Updated schema_info ({len(msg.content)} chars)")
            elif msg.name == "get_sample_data":
                updates["sample_data"] = msg.content
                logger.info(f"[tools] Updated sample_data ({len(msg.content)} chars)")
    
    return updates

# -------------------- Build Graph --------------------

# Initialize tools
tools = [
    get_database_schema,
    get_sample_data,
    generate_sql_with_context,
    quick_test_sql,
    query_existing_table,
]

# Initialize LLM with tools
llm = init_chat_model(
    MODEL_ID, 
    model_provider="bedrock_converse", 
    region_name=REGION
)
llm_with_tools = llm.bind_tools(tools)

# สร้าง checkpointer และ store สำหรับ AgentCore
checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)

# สร้าง graph
workflow = StateGraph(CustomState)
workflow.add_node("agent", call_model)

# เปลี่ยนจาก ToolNode(tools) เป็น custom function
from langgraph.types import CachePolicy

workflow.add_node(
    "tools",
    tools_with_state_update,
    cache_policy=CachePolicy(ttl=300)  # cache 5 นาที
)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")


# Compile พร้อม checkpointer และ store
app = workflow.compile(
    checkpointer=checkpointer,
    store=store, 
)


# -------------------- Main Execution --------------------

import time
from botocore.exceptions import ClientError


def get_previous_state(thread_id: str, actor_id: str):
    """Retrieve the previous state from checkpointer to preserve memory."""
    try:
        # Get the saved checkpoint state for this thread
        saved_state = app.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }
        )
        if saved_state and saved_state.values:
            return {
                "schema_info": saved_state.values.get("schema_info", ""),
                "sample_data": saved_state.values.get("sample_data", ""),
            }
    except Exception as e:
        logger.warning(f"Could not retrieve previous state: {e}")
    return {"schema_info": "", "sample_data": ""}


def invoke_agent(
    question: str,
    actor_id: str = "redshift-agent",
    session_id: str = "session-1",
    reset_memory: bool = False,
):
    config = {
        "configurable": {
            "thread_id": session_id,
            "actor_id": actor_id,
        }
    }
    
    logger.info("=" * 80)
    logger.info(f"User Question: {question}")
    logger.info(f"Actor: {actor_id} | Session: {session_id}")
    logger.info("=" * 80)
    
    # --- Load previous state from checkpointer ---
    if not reset_memory:
        prev_state = get_previous_state(session_id, actor_id)
        logger.info(f"[invoke] Loaded from checkpoint - schema: {len(prev_state['schema_info'])} chars, sample: {len(prev_state['sample_data'])} chars")
    else:
        prev_state = {"schema_info": "", "sample_data": ""}
        logger.info("[invoke] Reset memory - starting fresh")
    
    # --- Prepare payload with loaded state ---
    max_retries = 3
    payload = {
        "messages": [HumanMessage(content=question)],
        "schema_info": prev_state["schema_info"],
        "sample_data": prev_state["sample_data"],
    }

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[invoke] Calling app.invoke with config: {config}")
            result = app.invoke(payload, config=config)
            logger.info(f"[invoke] Result returned. Final state - schema: {len(result.get('schema_info', ''))} chars, sample: {len(result.get('sample_data', ''))} chars")
            break  # สำเร็จ
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if attempt == max_retries:
                    raise
                wait = (2 ** attempt) + 1  # 3s, 5s, 9s
                logger.warning(f"Throttling! รอ {wait} วินาที... (ครั้งที่ {attempt+1})")
                time.sleep(wait)
                
                # ← Reload state from checkpoint before retry to get any updates
                if attempt < max_retries:
                    reloaded_state = get_previous_state(session_id, actor_id)
                    payload["schema_info"] = reloaded_state["schema_info"]
                    payload["sample_data"] = reloaded_state["sample_data"]
                    logger.info(f"[retry] Reloaded state - schema: {len(payload['schema_info'])} chars, sample: {len(payload['sample_data'])} chars")
            else:
                raise
    else:
        raise RuntimeError("ไม่สามารถดำเนินการได้หลังจาก retry")

    # --- ดึง response ---
    final_message = result["messages"][-1]
    response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    logger.info("=" * 80)
    logger.info(f"Agent Response:\n{response_text}")
    logger.info("=" * 80)
    
    return response_text

# -------------------- Entry Point --------------------

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Text2SQL Agent with LangGraph + AWS AgentCore Memory")
    print("=" * 80)
    print("Features:")
    print("  ✅ Multi-turn conversations with memory")
    print("  ✅ Long-term memory persistence (AgentCore)")
    print("  ✅ Structured agent workflow (LangGraph)")
    print("  ✅ 5 Core tools for SQL generation & execution")
    print("  ✅ Pet Food Sales Database (Products + Transactions)")
    print("=" * 80)
    
    import uuid
    new_uuid = uuid.uuid4()

    # Example usage
    actor = "user-tanpat"
    session = str(new_uuid)
    
    # First question - Initial query
    print("\n💬 Question 1: Initial product query")
    response1 = invoke_agent(
        "Show me all dog food products with their prices",
        actor_id=actor,
        session_id=session
    )
    
    # Follow-up question - Multi-turn conversation
    print("\n" + "=" * 80)
    print("💬 Question 2: Follow-up with filter (testing multi-turn)")
    print("=" * 80)
    
    response2 = invoke_agent(
        "Now filter only products above 500 THB and sort by price descending",
        actor_id=actor,
        session_id=session
    )
    
    # Analytics question - Testing agent's analytical capabilities
    print("\n" + "=" * 80)
    print("💬 Question 3: Analytics query")
    print("=" * 80)
    
    response3 = invoke_agent(
        "What's the total revenue by store location in the last week?",
        actor_id=actor,
        session_id=session
    )