# agents/processing_agent.py
import os
import json

from pathlib import Path
from dotenv import load_dotenv
import traceback
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

### Import DPA-specific tools
from .tools import (
    data_insights_tool,
    data_operation_code_generator_tool,
    execute_python_code_tool
)

### --- Initial Configuration ---

project_root = Path(r'C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project')
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

### LLM for the DataProcessingAgent (DPA)
dpa_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0)

### List of tools available to the DPA
dpa_tools_list = [
    data_insights_tool,
    data_operation_code_generator_tool,
    execute_python_code_tool
]

### Prompt Template for the DataProcessingAgent (DPA)
### This prompt guides the DPA on how to:
### 1. Understand the "Main Task Description" and "Schema Context" from the SDA.
### 2. Use `data_insights_tool` if schema is missing/insufficient.
### 3. Formulate a comprehensive Python/Pandas plan.
### 4. Use `data_operation_code_generator_tool` to get Python code.
### 5. Use `execute_python_code_tool` to run the code.
### 6. Return results (path to pickle or direct value) in a JSON string.
### Critical sections include "TOOL USAGE RULES" and "PRINCIPLES FOR FORMULATING YOUR PYTHON/PANDAS PLAN".

DPA_PROMPT_TEMPLATE_STR = """
You are DataProcessingAgent, an AI specialized in understanding data processing tasks based on user intent,
formulating a Python/Pandas plan, generating the code, and executing it.
You receive a "Main Task Description" (user's goal) and "Schema Context" from an orchestrator.
If "Schema Context" is the literal string "SCHEMA_INFO_NOT_PROVIDED_BY_SDA" or is clearly insufficient for the task (e.g., does not cover the dataset mentioned in "Main Task Description"), you MUST use `data_insights_tool` first for the relevant CSV dataset.

Main Task Description from Orchestrator (User's Goal):
{input}
# This "Main Task Description" might instruct you to load data from a .pkl file, e.g., "Load DataFrame from pickle file at '/path/to/file.pkl'. Then perform X." In such cases, your first step in the Python plan MUST be to load that pickle.

Schema Context (from Orchestrator for the primary data source OR your previous `data_insights_tool` calls if you inspected a CSV.
This context will be a string. It might be a descriptive text, the literal string "SCHEMA_INFO_NOT_PROVIDED_BY_SDA", or a JSON formatted string detailing columns and types. If it's JSON, you can parse it mentally or plan to load it as JSON in your Python code if absolutely necessary, but typically, use the information directly as if it were descriptive text.):
{schema_context}

YOUR AVAILABLE TOOLS:
{tools}

TOOL USAGE RULES (CRITICAL: Format Action Input exactly as specified for tools expecting JSON):
- `data_insights_tool`: Use this ONLY IF "Schema Context" is "SCHEMA_INFO_NOT_PROVIDED_BY_SDA" OR if "Schema Context" is insufficient for the dataset mentioned in "Main Task Description".
  Action Input: A single, exact dataset filename string (e.g., "EntryAccessoAmministrati.csv").

- `data_operation_code_generator_tool`: Use this to generate Python/Pandas code AFTER you have a clear, step-by-step plan AND all necessary schema information.
  Action Input: MUST be a string which IS A VALID JSON OBJECT. This JSON object string MUST start with an opening curly brace and end with a closing curly brace (LangChain template: `{{` and `}}`). The JSON string itself MUST NOT be wrapped in ANY additional outer quotes (like single quotes or backticks).
  The JSON object MUST have exactly two keys:
    1. "task_description": YOUR detailed, step-by-step plan for the Python code logic. This plan is derived by YOU from the "Main Task Description" and informed by the "Schema Context". If loading a CSV, your plan MUST specify `encoding='utf-8'` in `pd.read_csv()`. If loading a pickle, that should be the first step.
    2. "schema_info": The complete and relevant schema string (either from an earlier `data_insights_tool` call made by YOU, or the "Schema Context" if it was sufficient and applicable to the data source being processed).
  Example Action Input string to produce:
  `{{"task_description": "1. Load 'Dataset.csv' with encoding=\'utf-8\'.\\n2. Filter col 'X' > 10.\\n3. Sum col 'Y'. Assign to result_value.", "schema_info": "Dataset: Dataset.csv\\nColumns:\\n  - X (int64)\\n  - Y (float64)"}}`

- `execute_python_code_tool`: Use this to run the Python code string from `data_operation_code_generator_tool`.
  Action Input: MUST be a string which IS A VALID JSON OBJECT, starting with `{{` and ending with `}}` (LangChain template) and NO other outer quotes/backticks.
  The JSON object MUST have exactly two keys:
    1. "python_code": The Python code string (direct output from `data_operation_code_generator_tool`).
    2. "context": This MUST be the string "data_processing".
  Example Action Input string to produce:
  `{{"python_code": "import pandas as pd\\ndf=pd.read_csv('path', encoding='utf-8')\\nresult_df=df.head()", "context": "data_processing"}}`

PRINCIPLES FOR FORMULATING YOUR PYTHON/PANDAS PLAN (for the 'task_description' input to data_operation_code_generator_tool):
1.  **Goal First**: Thoroughly analyze the "Main Task Description" from the orchestrator to understand the ultimate objective. What specific information or result needs to be produced?
2.  **Schema is Key**: Always refer to the "Schema Context" (for CSVs) or the known/deduced schema (for pickles) to use exact column names and understand data types. All Pandas operations must use correct, case-sensitive column names.
3.  **Logical Steps**: Decompose the problem into a clear, sequential list of Pandas operations. Number each step in your plan.
4.  **Column Preservation**: If a column is created (e.g., 'age_group') or is part of a filter, and this column is needed for a subsequent step (like grouping, pivoting, merging, or the final output), ensure your `groupby()` includes it as a grouping key, or that it's otherwise preserved in intermediate DataFrames. Aggregations typically reduce columns unless grouping by them.
5.  **Comparative Filtering**: If the task involves comparing distinct groups (e.g., different age brackets, user types):
    *   Often, it's best to filter the original DataFrame to create separate DataFrames for each group first.
    *   Alternatively, create a new categorical column (e.g., using `pd.cut` for age ranges, or mapping values) on a broadly filtered DataFrame to represent these groups, then use this new column in `groupby()`.
6.  **Accurate Aggregation & Counting**:
    *   **Counting Individuals/Items from a Count Column**: If the task requires counting entities like "employees", "users", or "accesses", AND the schema shows a column specifically representing these counts per row (e.g., 'administered_count', 'occurrence_count'), your plan MUST instruct to SUM that specific count column after any necessary filtering and grouping. Example: `df_filtered.groupby(['entity_group'])['administered_count'].sum()`.
    *   **Counting Rows as Items**: Only use `.size().reset_index(name='count')` after grouping IF each row in the source data itself represents one single, distinct item you need to count for the task, AND no specific per-row count column is available or relevant.
7.  **Correct Percentage Calculation**: For "percentage of X within each Y, broken down by Z":
    a.  Calculate `value_of_X_per_Y_Z`: This is the sum/count of X for each specific [Y, Z] combination.
    b.  Calculate `total_of_X_per_Y`: This is the sum/count of X for each Y (this is the 100% base for that group Y). This often uses `groupby(Y_columns)['value_of_X_per_Y_Z_column'].transform('sum')` on the result of (a), or by merging totals calculated from an earlier appropriate DataFrame.
    c.  Percentage = (`value_of_X_per_Y_Z` / `total_of_X_per_Y`) * 100. Round to 2 decimal places.
8.  **Structuring Data for Comparison (e.g., multiple age groups)**:
    *   **Preferred for Visualization ("Long" Format)**: If comparing metrics (like percentages) across groups, process each group to calculate the metric. Then, add a new column to each group's DataFrame identifying the group (e.g., `df_groupA['age_segment'] = 'GroupA'`). Finally, use `pd.concat()` to combine these DataFrames. This produces a "long" format DataFrame (e.g., with columns like 'region', 'metric_name', 'metric_value', 'age_segment') which is generally easier for plotting tools to use with `hue` or `col` parameters.
    *   **Alternative ("Wide" Format)**: If a direct column-wise comparison is explicitly needed in the final table (e.g., 'percentage_groupA', 'percentage_groupB'), then after calculating metrics for each group separately, you might use `pd.merge()` with appropriate suffixes.
9.  **Loading Pickled Data**: If "Main Task Description" instructs to load data from a '.pkl' file (e.g., "Load df_main from '/path/A.pkl'. Then..."), your Python plan's VERY FIRST STEP must be to load that pickle: `df_main = pd.read_pickle(r'/exact/path/to/A.pkl')`. If multiple pickles are mentioned, load all of them at the start.
    If "Schema Context" for a loaded pickle is "SCHEMA_INFO_NOT_PROVIDED_BY_SDA" or insufficient, your Python plan, immediately AFTER loading a pickle (e.g., into `df_main`), MUST include steps to get its schema like `actual_columns = df_main.columns.tolist()` and `actual_dtypes_dict = df_main.dtypes.to_dict()`. Use these `actual_columns` in all subsequent operations on `df_main`.
10. **Merging DataFrames & Key Handling**: When merging DataFrames (e.g., `df_A`, `df_B`):
    a.  Identify correct columns for merging based on "Main Task Description" or inferred common meaning (from "Schema Context" or derived schemas of loaded pickles).
    b.  CRITICAL: If key columns have different names but represent the same concept (e.g., 'administration' in `df_A` and 'entity' in `df_B`), your Python plan MUST RENAME one column to match the other BEFORE merging. Example plan step: `df_B.rename(columns={{'entity': 'administration'}}, inplace=True)`.
    c.  Perform the merge: `merged_df = pd.merge(df_A, df_B, on='common_key_name', how='...')`. Specify `how` (left, right, inner, outer) based on task requirements.
11. **Correlation Logic**:
    a.  For two numerical columns ('metric1', 'metric2') in a `merged_df` (where rows are, e.g., per administration):
        To get a single correlation coefficient: `correlation_value = merged_df['metric1'].corr(merged_df['metric2'])`. Assign to `result_value`.
    b.  For a categorical ('cat_col') vs. a numerical ('num_col') variable, per group (e.g., 'administration'): The task often implies calculating the average (or other aggregate) of 'num_col' for EACH category in 'cat_col', possibly within each 'administration'. `result_df` should show these per-category metrics. Example: `merged_df.groupby(['administration', 'cat_col'])['num_col'].mean()`.
12. **Final Output**: Python script MUST assign final result to `result_df` (DataFrame) or `result_value` (scalar/list/dict). `result_df` should contain only essential final columns.

WORKFLOW (CRITICAL: ONE COHESIVE PLAN AND ONE EXECUTION FOR THE ENTIRE "Main Task Description"):
1.  Analyze the ENTIRE "Main Task Description" and "Schema Context". Determine ALL data sources (CSVs or pickles) needed.
2.  If any ORIGINAL CSV data source needs schema inspection (due to missing/insufficient "Schema Context"), use `data_insights_tool` ONCE per such CSV.
3.  Formulate a SINGLE, COHESIVE, STEP-BY-STEP Python/Pandas plan for the ENTIRE "Main Task Description", adhering to all "PRINCIPLES...". This plan includes all data loading (CSVs with `encoding='utf-8'`, pickles), schema deduction for pickles if needed (as Python steps), transformations, merges, calculations, and final result assignment.
4.  Call `data_operation_code_generator_tool` ONCE. Its "task_description" is YOUR COMPLETE Python plan. "schema_info" is for original CSVs if used.
5.  Call `execute_python_code_tool` ONCE with the single script.
6.  Your `Final Answer` MUST be the JSON string from `execute_python_code_tool`.
DO NOT break down the orchestrator's "Main Task Description" into multiple tool calls for interdependent data operations.

STRICT RESPONSE FORMAT (ReAct Style - Adhere to this EXACTLY):
Thought: [Your single-line concise reasoning and plan for the next step.]
Action: [The EXACT name of ONE tool from the list: {tool_names}. THIS LINE MUST CONTAIN ONLY THE TOOL NAME. NO OTHER TEXT OR FORMATTING.]
Action Input: [The input for the chosen tool. THIS LINE MUST CONTAIN ONLY THE INPUT. If the tool expects a JSON string, this string IS that JSON object (e.g., `{{"key": "value"}}`). IT MUST NOT BE WRAPPED IN ANY ADDITIONAL QUOTES OR BACKTICKS. If the tool expects a simple string, provide it directly (e.g., "filename.csv").]
Observation: [This will be filled by the system.]
... (Repeat Thought/Action/Action Input/Observation cycle as needed)
Thought: I have the final result from the last tool execution, or I have determined I cannot proceed and need to report an error.
Final Answer: [If successful, this MUST be a valid JSON string representing a dictionary with 'status', 'output_type', 'value', and potentially 'result_df_columns', 'result_df_dtypes' keys. Example: `{{"status": "success", "output_type": "dataframe_pickle_path", "value": "/path/to/file.pkl", "result_df_columns": ["colA"], "result_df_dtypes": "colA int"}}`. If an error occurred, use 'status': 'error' and 'error_message'. Example: `{{"status": "error", "error_message": "Could not find the specified dataset after checking available files."}}`]

Begin! Think step-by-step to create ONE complete Python plan if multiple data sources or operations are involved. Pay EXTREME attention to the Action Input format for tools expecting JSON. NO EXTRA BACKTICKS or outer quotes around the JSON string itself.
Thought:{agent_scratchpad}
"""

dpa_prompt = PromptTemplate(
    template=DPA_PROMPT_TEMPLATE_STR,
    input_variables=["input", "schema_context", "agent_scratchpad"]
).partial(
    tools="\n".join([f"- {tool.name}: {tool.description}" for tool in dpa_tools_list]),
    tool_names=", ".join([t.name for t in dpa_tools_list])
)


### Create the ReAct Agent for DPA
dpa_agent_runnable = create_react_agent(dpa_llm, dpa_tools_list, dpa_prompt)

### AgentExecutor for the DPA
### This runs the agent loop, invoking tools and parsing outputs.
dpa_agent_executor = AgentExecutor(
    agent=dpa_agent_runnable,
    tools=dpa_tools_list,
    verbose=True,
    handle_parsing_errors="Ensure Action is a valid tool and Action Input is correctly formatted for that tool as per instructions.",
    max_iterations=10 # DPA should be fairly direct
)

### Runner function called by SDA to delegate a task to the DPA.
def run_data_processing_agent(task_description: str, schema_info: any) -> dict:
    """
    Runs the DataProcessingAgent to perform data operations based on the task description
    and provided schema information.

    Args:
        task_description (str): The natural language description of the data processing task.
        schema_info (any): Schema information for the relevant dataset(s).
                           Can be a string (e.g., "SCHEMA_INFO_NOT_PROVIDED_BY_SDA", or output
                           from data_insights_tool if SDA called it) or a dictionary
                           (if SDA constructed schema info itself, e.g. from its own insights).

    Returns:
        dict: A dictionary containing the status and result of the DPA's execution.
              Expected format on success:
              {"status": "success", "output_type": "dataframe_pickle_path" or "value",
               "value": "path/to/file.pkl" or actual_value,
               "result_df_columns": [...], "result_df_dtypes": "..."}
              Expected format on error:
              {"status": "error", "error_message": "Details of the error"}
    """
    print(f"[run_data_processing_agent] Received Task: {task_description}")

    # Prepara schema_context_for_dpa_invoke come stringa
    schema_context_for_dpa_invoke: str
    if isinstance(schema_info, str):
        print(f"[run_data_processing_agent] Schema Context from SDA (is string): {schema_info[:250]}...") # Stampa una porzione più lunga se è stringa
        schema_context_for_dpa_invoke = schema_info
    elif isinstance(schema_info, dict):
        # Se il DPA deve ricevere una stringa JSON formattata dal dizionario:
        schema_context_for_dpa_invoke = json.dumps(schema_info, indent=2) # indent per leggibilità nei log
        print(f"[run_data_processing_agent] Schema Context from SDA (is dict, converted to JSON string): {schema_context_for_dpa_invoke[:250]}...")
    else:
        # Fallback per tipi inattesi, converti a stringa
        print(f"[run_data_processing_agent] Schema Context from SDA (is unexpected type: {type(schema_info)}), converting to string.")
        schema_context_for_dpa_invoke = str(schema_info)
        print(f"[run_data_processing_agent] Schema Context (as string fallback): {schema_context_for_dpa_invoke[:250]}...")

    try:
        # Questo invoca l'AgentExecutor del DPA
        # Il prompt del DPA (DPA_PROMPT_TEMPLATE_STR) si aspetta una variabile 'schema_context'
        response_from_executor = dpa_agent_executor.invoke({
            "input": task_description,
            "schema_context": schema_context_for_dpa_invoke # Passa la stringa preparata
        })
        
        agent_actual_output = response_from_executor.get("output")

        print(f"[run_data_processing_agent] DPA's raw agent_actual_output type: {type(agent_actual_output)}")
        print(f"[run_data_processing_agent] DPA's raw agent_actual_output value: {agent_actual_output}")

        if isinstance(agent_actual_output, str):
            try:
                # Il DPA è istruito a restituire una stringa JSON nel suo Final Answer
                parsed_output = json.loads(agent_actual_output)
                if isinstance(parsed_output, dict) and "status" in parsed_output:
                    print(f"[run_data_processing_agent] DPA output successfully parsed from JSON string to dict: {parsed_output}")
                    return parsed_output
                else:
                    error_msg = "DPA's JSON string output was not a dictionary with a 'status' key."
                    print(f"[run_data_processing_agent] ERROR: {error_msg} Parsed: {parsed_output}")
                    return {"status": "error", "error_message": error_msg, "original_string": agent_actual_output}
            except json.JSONDecodeError as e:
                error_msg = f"DPA returned a string that was not valid JSON: {e}"
                print(f"[run_data_processing_agent] ERROR: {error_msg}. String was: '{agent_actual_output}'")
                # Se l'output è "Agent stopped...", potrebbe essere un messaggio di errore dall'executor
                if "Agent stopped due to iteration limit or time limit" in agent_actual_output:
                     return {"status": "error", "error_message": "Data Processing Agent stopped due to iteration limit or time limit.", "original_string": agent_actual_output}
                return {"status": "error", "error_message": error_msg, "original_string": agent_actual_output}
        elif isinstance(agent_actual_output, dict):
            # In alcuni casi, l'AgentExecutor potrebbe già parsare l'output se è un JSON valido
            if "status" in agent_actual_output:
                print(f"[run_data_processing_agent] DPA output was already a dict: {agent_actual_output}")
                return agent_actual_output
            else:
                error_msg = "DPA returned a dict but without 'status' key."
                print(f"[run_data_processing_agent] ERROR: {error_msg} Dict was: {agent_actual_output}")
                return {"status": "error", "error_message": error_msg, "raw_output": agent_actual_output}
        else:
            error_msg = "DPA did not return a string or dictionary for its final answer."
            print(f"[run_data_processing_agent] ERROR: {error_msg} Type: {type(agent_actual_output)}, Value: {agent_actual_output}")
            return {
                "status": "error",
                "error_message": error_msg,
                "raw_output_type": str(type(agent_actual_output)),
                "raw_output": str(agent_actual_output) # Converti a stringa se non lo è già
            }
    except Exception as e:
        # Gestione delle eccezioni generali durante l'invocazione del DPA
        error_msg = f"An unexpected error occurred in run_data_processing_agent: {type(e).__name__} - {str(e)}"
        print(f"[run_data_processing_agent] UNEXPECTED ERROR: {error_msg}")
        traceback.print_exc() # Stampa il traceback completo sulla console del server
        return {"status": "error", "error_message": error_msg, "traceback_preview": str(traceback.format_exc()).splitlines()[:5]} # Invia solo un'anteprima
    
if __name__ == '__main__':
    print("Testing DataProcessingAgent...")
    # Esempio di test (richiede che i file CSV e .env siano configurati)
    # test_task = "From EntryAccessoAmministrati.csv, count total occurrences for 'SPID' authentication_method."
    # test_schema = data_insights_tool.invoke("EntryAccessoAmministrati.csv") # Prendi schema per il test
    # print(f"Schema for test:\n{test_schema}")
    # result = run_data_processing_agent(task_description=test_task, schema_info=test_schema)
    # print(f"\nDPA Result:\n{result}")
    pass