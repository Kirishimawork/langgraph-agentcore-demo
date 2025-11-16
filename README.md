## 💡 Text2SQL Agent Using LangGraph & AWS Bedrock AgentCore

### What is this?

This code implements a **Text-to-SQL Agent** built with **LangGraph**, a framework for creating stateful, resilient agents. It functions as a Data Analyst Assistant, capable of connecting to an **Amazon Redshift** data warehouse and leveraging Large Language Models (LLMs) from **AWS Bedrock** (e.g., Claude 3 Sonnet and Amazon Nova Pro).

Its standout feature is the integration with **AWS AgentCore Memory**. This component acts as a checkpointer to provide the agent with **long-term persistent memory**, allowing it to remember conversational context and cached data even after the session ends.

### What does it do? (How?)

The agent operates as a **LangGraph State Machine**, following a defined cycle to convert a user's natural language question into a precise database-driven answer.

1.  **Receive Question:** The user provides a prompt (e.g., "What were the total sales for dog food last week?").
2.  **Check Cache:** The agent (driven by the `call_model` node) inspects its `CustomState` to see if the database schema (`schema_info`) or `sample_data` is already cached in memory from a previous turn.
3.  **Fetch Context (If needed):**
    * If the cache is empty, the agent invokes the `get_database_schema` and `get_sample_data` tools to query Redshift for this structural information.
    * The custom `tools_with_state_update` function then **caches these results** directly into the `CustomState` for future use.
4.  **Generate SQL:** With the schema context available, the agent uses the `generate_sql_with_context` tool (powered by a specialized LLM like Amazon Nova Pro) to create the appropriate SQL query.
5.  **Validate SQL:** It runs the `quick_test_sql` tool to ensure the generated query is syntactically valid before executing a potentially expensive query.
6.  **Execute & Fetch:** Once validated, the `query_existing_table` tool runs the final query against Redshift to retrieve the actual data.
7.  **Summarize & Respond:** The main agent LLM (Claude 3 Sonnet) formats the raw data results into a clear, human-readable answer for the user.

### Why does this exist?

This architecture is designed to solve two critical challenges in database-querying agents:

1.  **Efficiency (Solving Redundant Fetches):** Fetching database schema on every single query is slow, costly (in both time and API calls), and highly redundant.
    * **Solution:** This agent uses **State Caching**. By storing `schema_info` and `sample_data` in its `CustomState`, the agent fetches this information from Redshift **only once**. All subsequent follow-up questions are **significantly faster** and more cost-effective because they reuse the cached context.

2.  **Persistence (Solving "Session Amnesia"):** Standard agents forget their state and all cached data the moment a session ends or the application restarts.
    * **Solution:** This agent uses **AWS AgentCore Memory** as a persistent checkpointer. It automatically saves the `CustomState` (including the valuable cached schema) to a durable, long-term store. This allows a user to disconnect, return hours or days later, and **resume the conversation exactly where they left off**, with the agent instantly recalling the database context.

> **In summary:** This is a high-efficiency Text-to-SQL agent designed for a robust, real-world application. It enables non-technical users to "chat" with their Redshift data warehouse, providing a fast, persistent, and conversational experience, much like having a dedicated data analyst on standby.
