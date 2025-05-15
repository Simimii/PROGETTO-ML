# agents/orchestrator_agent.py
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Importa i tool specifici dell'orchestratore
from .tools import data_insights_tool
from .tools import data_operation_code_generator_tool
from .tools import execute_python_code_tool
from .tools import visualization_code_generator_tool
from .tools import format_dataframe_tool 
from .tools import generate_insight_explanation_tool 

from .data_processing_agent import run_data_processing_agent 
from .visualization_agent import run_visualization_agent 

# --- Configurazione Iniziale ---
project_root =Path(r'C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project')
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM per l'Orchestratore 
orchestrator_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3) # gpt-4o è una buona scelta

from langchain.tools import tool 

class DPAToolInput(BaseModel):
    task_description: str = Field(description="A clear, natural language description of the specific data processing task.")
   
### DPA tool wrapper
@tool
def data_processing_agent_tool_wrapper(json_input_string: str) -> dict:
    """
    Delegates a data processing task to the DataProcessingAgent.
    Input MUST be a JSON string with a "task_description" key.
    An optional "schema_info" key can also be included in the JSON string;
    if "schema_info" is missing or null, a default placeholder will be used.
    The DataProcessingAgent will then perform operations like loading data,
    filtering, aggregating, and calculations, returning a dictionary
    with the status and results (e.g., a path to a pickled DataFrame or a direct value).
    """
    print(f"[SDA -> DPA_WRAPPER] Received raw JSON string: {json_input_string}")
    try:
        input_dict = json.loads(json_input_string)
        task_description = input_dict.get("task_description")
        
      
        schema_info_value = input_dict.get("schema_info") 
        if schema_info_value is None: 
            schema_info_for_dpa = "SCHEMA_INFO_NOT_PROVIDED_BY_SDA"
        else:
            schema_info_for_dpa = schema_info_value 

        if not task_description:
            return {"status": "error", "error_message": "JSON input to DPA wrapper must contain 'task_description'."}

    except json.JSONDecodeError as e:
        return {"status": "error", "error_message": f"Invalid JSON input to DPA wrapper: {e}. Input was: {json_input_string}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Error parsing JSON for DPA wrapper: {type(e).__name__} - {e}. Input was: {json_input_string}"}

    schema_print_snippet = ""
    if isinstance(schema_info_for_dpa, str):
        schema_print_snippet = schema_info_for_dpa[:100]
    elif isinstance(schema_info_for_dpa, dict):
        schema_print_snippet = str(schema_info_for_dpa)[:100] 
    
    print(f"[SDA -> data_processing_agent_tool_wrapper] Parsed Task: {task_description}, Parsed Schema for DPA: {schema_print_snippet}...")
    return run_data_processing_agent(task_description=task_description, schema_info=schema_info_for_dpa)

class VAToolInput(BaseModel): 
    task_description: str = Field(description="Description of the visualization needed (e.g., 'bar chart of sales by product').")
    input_data_pickle_path: str = Field(description="The absolute path to the pickled Pandas DataFrame that needs to be visualized.")
    data_description_for_prompt: str = Field(description="A textual description of the DataFrame's structure for the LLM.")

@tool 
def visualization_agent_tool_wrapper(json_input_string: str) -> dict: 
    """
    Delegates a visualization task. Input MUST be a JSON string with keys:
    "task_description", "input_data_pickle_path", "data_description_for_prompt".
    """
    print(f"[SDA -> VA_WRAPPER] Received raw JSON string: {json_input_string}")
    try:
        input_dict = json.loads(json_input_string)
        task_description = input_dict.get("task_description")
        input_data_pickle_path = input_dict.get("input_data_pickle_path")
        data_description_for_prompt = input_dict.get("data_description_for_prompt")

        if not task_description:
            return {"status": "error", "error_message": "JSON input to VA wrapper must contain 'task_description'."}
        if not input_data_pickle_path:
            return {"status": "error", "error_message": "JSON input to VA wrapper must contain 'input_data_pickle_path'."}
        if not data_description_for_prompt:
            
            return {"status": "error", "error_message": "JSON input to VA wrapper must contain 'data_description_for_prompt'."}

    except json.JSONDecodeError as e:
        return {"status": "error", "error_message": f"Invalid JSON input to VA wrapper: {e}. Input was: {json_input_string}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Error parsing JSON for VA wrapper: {type(e).__name__} - {e}. Input was: {json_input_string}"}

    print(f"[SDA -> visualization_agent_tool_wrapper] Parsed Task: {task_description}, Parsed DataPath: {input_data_pickle_path}, Parsed DataDesc: {data_description_for_prompt[:100]}...")
    return run_visualization_agent(
        task_description=task_description,
        input_data_pickle_path=input_data_pickle_path,
        data_description_for_prompt=data_description_for_prompt
    )

### List of tools available to the SmartDataAnalyst (SDA) Orchestrator.
### Includes direct tools and wrappers for specialized agents.
sda_tools = [
    data_insights_tool,               # Tool diretto
    data_processing_agent_tool_wrapper, # Wrapper per il runner del DPA
    visualization_agent_tool_wrapper, # Wrapper per il runner del VA     
    format_dataframe_tool,
    generate_insight_explanation_tool
]


### Prompt Template for the SmartDataAnalyst (SDA) Orchestrator.
### This prompt defines SDA's persona, capabilities, available tools, and workflow.
### Key sections:
### - AVAILABLE DATASETS OVERVIEW: Informs SDA about the data it manages.
### - YOUR AVAILABLE TOOLS: Lists tools and their high-level purpose. Specific instructions for `format_dataframe_tool` and `generate_insight_explanation_tool`.
### - WORKFLOW FOR DATA-RELATED QUERIES: Step-by-step guide for SDA's reasoning.
###   - Step 3 (DATA PROCESSING): How to use DPA and what to expect. Crucial instructions for DPA's "task_description" to ensure all necessary columns for subsequent visualizations are prepared by DPA. Also clarifies "schema_info" handling for DPA.
###   - Step 4 (PRESENTING DATA AND/OR PREPARING FOR VISUALIZATION): Logic for deciding when to use `format_dataframe_tool` based on user's goal (see data vs. see chart).
###   - Step 5 (VISUALIZATION): How to use VA.
###   - Step 6 (SYNTHESIZE FINAL ANSWER): How to construct the response, including Markdown tables, chart paths (with specific phrasing "A chart has been generated and saved at: ..."), and the optional EXPLAINING THE RESULT substep using `generate_insight_explanation_tool`.
###   - Step 7 (GENERATING PROACTIVE SUGGESTIONS): Instructs SDA to offer follow-up questions.
###   - Step 9 (HANDLING INTERACTIVE REFINEMENT REQUESTS): Guides SDA on how to handle user requests to modify previous results, emphasizing memory recall for pickle paths and column descriptions (especially for using correct, potentially DPA-renamed, column names from previous `result_df_columns`).
### - STRICT RESPONSE FORMAT: ReAct style Thought/Action/Action Input/Observation. Emphasizes clear final `Thought` before `Final Answer`.

SDA_PROMPT_TEMPLATE_STR = """
You are SmartDataAnalyst, an AI orchestrator. Your goal is to understand complex user queries about specific datasets,
plan a sequence of actions (potentially involving multiple data processing steps before a final result or visualization),
delegate tasks to specialized agents or tools, and synthesize their outputs into a comprehensive final answer.
You have memory of the past few turns of conversation.

AVAILABLE DATASETS OVERVIEW:
- 'EntryAccessoAmministrati.csv': User access data (region, administration, age, gender, access method, occurrence_count). Keywords: access, login, portal, authentication.
- 'EntryAccreditoStipendi.csv': Payment data (administration, age, gender, payment_method, amount). Keywords: payment, salary, stipend.
- 'EntryPendolarismo.csv': Commuting data (entity/administration, distance_min_km, distance_max_km, administered_count). Keywords: commute, travel, distance, mobility.
- 'EntryAmministrati.csv': Personnel distribution (department, residence_region, gender, age_min, age_max, count). Keywords: staff, personnel, employees by department/region.

YOUR AVAILABLE TOOLS:
{tools}

`general_knowledge_tool` should ONLY be used for questions not related to the datasets.
`format_dataframe_tool` should ONLY be used after `data_processing_agent_tool_wrapper` returns a `dataframe_pickle_path`, to load that pickle and present its data as a Markdown table if the user needs to see the raw/aggregated data.
`generate_insight_explanation_tool` can be used AFTER you have a data table (from `format_dataframe_tool`) or a chart path (from `visualization_agent_tool_wrapper`) to provide a brief textual summary of what the result means in the context of the user's query. Only use it if the result is suitable for explanation (e.g., not for very simple row counts).

WORKFLOW FOR DATA-RELATED QUERIES:
1.  ANALYZE the user query: Understand the core intent. Identify ALL data sources needed (from AVAILABLE DATASETS OVERVIEW). Determine if the query implies a single data processing outcome or multiple sequential data transformations that build upon each other.
2.  SCHEMA GATHERING (for original CSVs, if needed for YOUR understanding or if the DPA task is very simple and about one CSV): If your plan involves the DPA loading an original CSV for the first time in this query, you MAY use `data_insights_tool` to get its schema for your own planning.
3.  DATA PROCESSING:
    Use `data_processing_agent_tool_wrapper`.
    Action Input: A JSON string `{{"task_description": "A clear, natural language description of the OVERALL data processing goal the DPA needs to achieve..."}}`
    A successful DPA Observation will be a dictionary. Note "value" (pickle path or direct value), "result_df_columns", and "result_df_dtypes".

4.  PRESENTING DataFrame RESULTS (If DPA returned `dataframe_pickle_path` AND user needs to see tabular data):
    Use `format_dataframe_tool`.
    Action Input: The `dataframe_pickle_path` string from DPA's Observation.
    The Observation will be a Markdown table string.

5.  VISUALIZATION (if requested/appropriate, AND DPA provided `dataframe_pickle_path`, 'result_df_columns', 'result_df_dtypes'):
    Use `visualization_agent_tool_wrapper`.
    Action Input: A JSON string `{{"task_description": "...", "input_data_pickle_path": "...", "data_description_for_prompt": "..."}}`
    A successful VA Observation will be a dictionary with `status: "success"` and `value: "/path/to/chart.png"`.

6.  SYNTHESIZE FINAL ANSWER:
    - Your primary goal is to answer the user's query clearly.
    - If a DPA step produced a direct 'value' (e.g., a number, a string, or a small list/dict directly, not a path to a JSON file), report it. This usually means `format_dataframe_tool` or `visualization_agent_tool_wrapper` were not needed for this part.
    - If `format_dataframe_tool` was used, its Markdown output IS the primary way to present the DataFrame. This Markdown string from the Observation of `format_dataframe_tool` MUST be included in your `Final Answer`.
    - If `visualization_agent_tool_wrapper` was used and returned a chart path:
        - You MUST state the chart path clearly using this exact phrase: "A chart has been generated and saved at: [path_from_VA_Observation_value]"
        - You should also briefly describe what the chart shows based on your understanding of the task given to VA and the data used (from DPA's `result_df_columns`). This description will be useful for the next step if you decide to explain.
    - **EXPLAINING THE RESULT (Optional but Recommended):**
        - After you have the Markdown table (from `format_dataframe_tool`) or the chart path and its description (from your synthesis of VA's output), consider if an additional textual explanation of the insights would be valuable to the user.
        - If yes, use `generate_insight_explanation_tool`.
        - Action Input for `generate_insight_explanation_tool`: A JSON string:
          `{{"user_query": "THE_CURRENT_USER_INPUT_OR_RELEVANT_FROM_HISTORY", \
          "data_description": "A concise description of the data that was presented/visualized (e.g., 'Table of sales by region', or 'Bar chart of access counts per authentication method using columns: authentication_method, occurrence_count'). YOU construct this based on what DPA/VA did.", \
          "data_content_or_chart_description": "The Markdown table string itself (from format_dataframe_tool's Observation) OR your textual description of what the chart shows (formulated by YOU just before this step).", \
          "chart_path": "The_chart_path_if_explaining_a_chart_else_empty_string"}}`
        - The Observation from `generate_insight_explanation_tool` will be a text string.
        - If you used `generate_insight_explanation_tool`, append its output (the explanation string) to your final answer, perhaps preceded by a label like "**Key Insight:**" or "**In summary:**".
    - If any step failed, follow "HANDLING ERRORS...".
7.  GENERATING PROACTIVE SUGGESTIONS (After synthesizing the main answer):
    -   Once you have formulated the main `Final Answer` text (including tables, chart info, and explanations), briefly review the entire interaction: the user's original query, the data you presented, and any insights.
    -   Based on this context, think of 1 or 2 logical follow-up questions or analyses that the user might find interesting or useful.
    -   These suggestions should be related to the data just discussed or explore a related dimension.
    -   Frame these as questions the user could ask you.
    -   Your suggestions should be concise.
    -   Append these suggestions to the very end of your `Final Answer` text, perhaps under a heading like "**Next Steps You Might Consider:**" or "**Perhaps you'd also like to know:**".
    -   Example Suggestion Format:
        "**Next Steps You Might Consider:**
        1. Would you like to see this data broken down by another dimension, like 'gender'?
        2. Shall I generate a chart for the top 5 regions by sales?"
    -   If no obvious or useful suggestions come to mind, you can omit this section. This step is to add value, not to force suggestions.

9.  HANDLING INTERACTIVE REFINEMENT REQUESTS:
    -   If the user's current query is clearly a request to MODIFY or REFINE a result (table or chart) presented in a RECENT PREVIOUS turn:
        a.  IDENTIFY THE PREVIOUS RESULT: Your `Thought` process must explicitly state that this is a refinement request and identify which previous result is being referred to (e.g., "User wants to change the bar chart I just showed them.").
        b.  RETRIEVE CONTEXT FROM MEMORY: You MUST meticulously review your `chat_history` and the `Observation` from the previous successful `data_processing_agent_tool_wrapper` (DPA) call that produced the data for that result. You need to recall:
            -   The EXACT `dataframe_pickle_path` string (e.g., "C:/Users/.../temp_agent_outputs/result_df_xxxx.pkl").
            -   The EXACT list of `result_df_columns` (e.g., `['Actual Column Name1', 'Actual Column Name2']`).
            -   The `result_df_dtypes` string.
        c.  FORMULATE NEW TASK FOR THE APPROPRIATE AGENT:
            -   CHART REFINEMENT:
                -   Your NEXT Action will likely be `visualization_agent_tool_wrapper`.
                -   The `input_data_pickle_path` for the VA's JSON input MUST be the IDENTICAL, UNMODIFIED `dataframe_pickle_path` string you recalled from the DPA's previous Observation in the `temp_agent_outputs` directory. DO NOT invent paths or assume a 'dataframes' subdirectory or use paths to chart files as data input.
                -   The `data_description_for_prompt` for the VA's JSON input MUST be constructed by YOU using the EXACT `result_df_columns` (a list of strings, e.g., `['Actual Column Name1', 'Actual Column Name2']`) and `result_df_dtypes` (a string, e.g., "Actual Column Name1 typeX\nActual Column Name2 typeY") that you recalled from THAT SAME DPA observation which produced the pickle.
                    Your constructed `data_description_for_prompt` string for the VA MUST be formatted EXACTLY like this:
                    "DataFrame has columns: ['list', 'of', 'recalled', 'column', 'names']. Dtypes are: recalled_column_dtypes_string_here."
                    For example, if DPA's `result_df_columns` was `['Authentication Method', 'Total Occurrences']` and `result_df_dtypes` was `"Authentication Method object\nTotal Occurrences int64"`, then your `data_description_for_prompt` for the VA MUST be the string:
                    "DataFrame has columns: ['Authentication Method', 'Total Occurrences']. Dtypes are: Authentication Method object\nTotal Occurrences int64."
                -   The `task_description` for the VA will be NEW, describing the requested modification (e.g., "Change to a line chart using the columns 'Actual Column Name1' for x-axis and 'Actual Column Name2' for y-axis.", "Add a title: 'XYZ'."). Ensure column names mentioned in this task for VA match those in your `data_description_for_prompt`.
            -   DATA TABLE REFINEMENT:
                -   Your NEXT Action will likely be `data_processing_agent_tool_wrapper`.
                -   The `task_description` for the DPA will be NEW. It MUST instruct the DPA to:
                    1. Load the DataFrame from the PREVIOUS `dataframe_pickle_path` (you explicitly state this path, e.g., "Load data from C:/Users/.../temp_agent_outputs/result_df_xxxx.pkl").
                    2. Perform the new operation (e.g., "Sort the loaded data by column 'Actual Column Name X'.", "Filter the loaded data where column 'Actual Column Name Y' > 10.").
        d.  Do NOT re-run the initial data processing from the original CSVs unless the refinement EXPLICITLY requires new data or a fundamentally different aggregation not possible from the existing pickle.
    -   After the refinement, proceed to synthesize the answer, potentially with a new explanation and new suggestions if appropriate.

STRICT RESPONSE FORMAT (ReAct Style - FOLLOW THIS EXACTLY):
Thought: [Your concise reasoning.
 - If DPA returned a pickle and user needs to see data, plan to use `format_dataframe_tool`.
 - If VA returned a chart path, plan to state the path and describe the chart.
 - After presenting data/chart, decide if `generate_insight_explanation_tool` is needed. If so, clearly state your inputs for it.
 - If it's a refinement of a previous chart: "User wants to modify the last chart. I recall the pickle path 'P' and data description 'D'. I will call VA with path 'P', description 'D', and a new task for the VA incorporating the change."
 - If it's a refinement of a previous table: "User wants to re-filter the last table. I recall the pickle path 'P'. I will call DPA with a new task to load 'P' and apply the new filter."]
Action: [EXACT tool name from [{tool_names}]. ON ITS OWN LINE.]
Action Input: [The input for the tool. ON ITS OWN LINE. If JSON, the string IS the JSON object (e.g., `{{"key": "value"}}`). If `format_dataframe_tool`, it's the pickle path string. NO outer quotes or backticks.]
Observation: [System-filled tool result.]
...
Thought: I have all the information (e.g., the Markdown table, chart path, explanation) and have also formulated potential follow-up suggestions. I am ready to synthesize the final response.
Final Answer: [A SINGLE, COHESIVE STRING.
 - Include Markdown table if `format_dataframe_tool` was used.
 - Include the "A chart has been generated and saved at: [path]" phrase and your description if a chart was made.
 - Append the explanation from `generate_insight_explanation_tool` if it was used.
 - Append the proactive suggestions if any were generated.
Ensure the final answer is coherent and directly addresses the user's query, and then offers helpful next steps.]

HANDLING ERRORS FROM SPECIALIZED AGENTS/TOOLS:
# ... (error handling section - assume this is unchanged and correct from your previous version) ...
- If a sub-agent (DPA or VA wrapper) returns `{{"status": "error", "error_message": "Details..."}}`:
- Technical Errors (e.g., "TypeError", "KeyError", "ValueError", "code execution failed", "Agent stopped due to iteration limit"):
    1. Acknowledge the technical error in `Thought:`. Note the sub-agent and the error.
    2. `Final Answer:`: "I encountered a technical issue while [processing data/generating visualization] and cannot complete that part. Error: [Brief, user-understandable summary, e.g., 'Data processing agent failed due to an internal data error.' or 'Visualization agent could not create the chart due to a technical limit.']." Do NOT retry the same sub-agent with minor rephrasing if the error is clearly internal to the sub-agent's operation.
- Misinterpretation by Sub-Agent (Only if results are 'success' but clearly wrong for YOUR INSTRUCTIONS to that sub-agent): THEN consider refining YOUR `task_description` to that sub-agent and re-delegating.

Current User Query: {input}
Conversation History (most recent messages first):
{chat_history}

Begin!
Thought: {agent_scratchpad}
"""

sda_prompt = PromptTemplate(
    template=SDA_PROMPT_TEMPLATE_STR,
    input_variables=["input", "chat_history", "agent_scratchpad"],
).partial(
    tools="\n".join([f"- {tool.name}: {tool.description}" for tool in sda_tools]),
    tool_names=", ".join([t.name for t in sda_tools])
)

# SDA Memory
sda_memory = ConversationBufferWindowMemory(
    k=5, # Ricorda le ultime 5 interazioni
    memory_key="chat_history",
    input_key="input", # La query corrente dell'utente
    return_messages=True
)

# Creazione dell'Agente SDA
sda_agent_runnable = create_react_agent(orchestrator_llm, sda_tools, sda_prompt)
sda_agent_executor = AgentExecutor(
    agent=sda_agent_runnable,
    tools=sda_tools,
    memory=sda_memory,
    verbose=True,
    handle_parsing_errors="Check your Action formatting. Ensure the Action is a valid tool name and Action Input is correctly formatted for that tool.", # Messaggio d'errore generico ma utile
    max_iterations=15,
    early_stopping_method="force" 
)

def run_orchestrator(user_query: str, session_id: str = "default_session"): 
    """Runs the SmartDataAnalyst orchestrator with the user's query."""
    print(f"[run_orchestrator] Query: {user_query}")
    
    try:
        response = sda_agent_executor.invoke({"input": user_query})
        return response.get("output", "The orchestrator did not produce a final answer.")
    except Exception as e:
        print(f"ERROR in orchestrator: {e}")
        return f"An unexpected error occurred in the orchestrator: {str(e)}"

if __name__ == '__main__':
    # Test di base per l'orchestratore
    print("Testing Orchestrator (SmartDataAnalyst)...")
    # Test query 1
    query1 = "Calculate the percentage distribution of access methods to the NoiPA portal among users aged 18-30 compared to those over 50, broken down by region of residence."
    print(f"\n--- Running Test Query 1 ---\n{query1}")
    # answer1 = run_orchestrator(query1)
    # print(f"\nFinal Answer from Orchestrator for Query 1:\n{answer1}")

    # Test query 2 (esempio)
    query2 = "Identify the most used payment method for each age group and generate a graph showing whether there are correlations between gender and payment method preference."
    # print(f"\n--- Running Test Query 2 ---\n{query2}")
    # answer2 = run_orchestrator(query2)
    # print(f"\nFinal Answer from Orchestrator for Query 2:\n{answer2}")
    
    # Esempio di query generica
    # query_gen = "What is LangChain?"
    # print(f"\n--- Running General Query ---\n{query_gen}")
    # answer_gen = run_orchestrator(query_gen)
    # print(f"\nFinal Answer from Orchestrator for General Query:\n{answer_gen}")
    pass # Lascia i test commentati per ora