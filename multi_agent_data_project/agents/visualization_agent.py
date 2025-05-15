# agents/visualization_agent.py
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import json

### Import VA-specific tools
from .tools import (
    visualization_code_generator_tool,
    execute_python_code_tool
)

### --- Initial Configuration ---
project_root = Path(r'C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project')
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

### LLM for the VisualizationAgent (VA)
va_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0)

### List of tools available to the VA
va_tools_list = [
    visualization_code_generator_tool,
    execute_python_code_tool 
]


### Prompt Template for the VisualizationAgent (VA)
### This prompt guides the VA on how to:
### 1. Use "input_data_path" (pickle from DPA) and "input_data_meta_description" (schema).
### 2. Give ABSOLUTE PRIORITY to "input_data_meta_description" for column names.
### 3. Formulate a plan for `visualization_code_generator_tool` (VCGT).
### 4. Call `execute_python_code_tool` to run the plot script.
### 5. Return the path to the generated image file.
### Critical sections: "TOOL USAGE RULES", "WORKFLOW AND CRITICAL CONSIDERATIONS", and "HANDLING ERRORS".

VA_PROMPT_TEMPLATE_STR = """
You are VisualizationAgent, an AI specialized in generating Python code for data visualizations (using Matplotlib/Seaborn)
and executing that code to produce chart image files.
You receive from an orchestrator (SmartDataAnalyst):
- "Main Task Description for Visualization": What the user wants to see plotted.
- "Path to Pickled DataFrame for Plotting": The .pkl data file to use. This will be loaded as `df` for your code.
- "Description of the DataFrame to be Plotted": THIS IS YOUR ABSOLUTE SOURCE OF TRUTH FOR THE DATAFRAME'S COLUMN NAMES, THEIR DATA TYPES, AND THE DATAFRAME'S OVERALL STRUCTURE. YOU MUST USE THESE EXACTLY.

Path to Pickled DataFrame for Plotting (this file will be loaded into the `df` variable for your Python code by the execute_python_code_tool. THIS IS THE EXACT PATH YOU MUST USE):
{input_data_path}

# visualization_agent.py - VA_PROMPT_TEMPLATE_STR

# ... (inizio del prompt VA) ...

Path to Pickled DataFrame for Plotting (this file will be loaded into the `df` variable for your Python code by the execute_python_code_tool. THIS IS THE EXACT PATH YOU MUST USE):
{input_data_path}

Description of the DataFrame `df` that will be loaded (CRITICAL: THIS IS YOUR ABSOLUTE, UNQUESTIONABLE, AND ONLY SOURCE OF TRUTH FOR THE DATAFRAME'S EXACT COLUMN NAMES...):
{input_data_meta_description}
# Example content for {input_data_meta_description}: "DataFrame has columns: ['entity_name', 'percentage_value', 'category_X']. Dtypes are: entity_name object, percentage_value float64, category_X object."

Main Task Description for Visualization (User's Goal for the Chart - adapt your plot to achieve this using ONLY the available columns from "Description of the DataFrame `df`" above):
{input_task_description}
# Example content for {input_task_description}: "Create a bar chart of 'Total Revenue' by 'Actual Sales Region'."

YOUR AVAILABLE TOOLS:
{tools}

TOOL USAGE RULES (CRITICAL: Format Action Input exactly as specified, using the provided input variables {input_data_path} and {input_data_meta_description} precisely):
- `visualization_code_generator_tool`: Use this to generate Python plotting code.
  Action Input: MUST be a string which IS A VALID JSON OBJECT (starts with `{{`, ends with `}}`, no outer quotes or backticks).
  This JSON object MUST have THREE keys:
    1. "task_description": YOUR detailed plan for the plot. ALL column names used in this plan (for x-axis, y-axis, hue, col/facet, `melt` parameters like `id_vars`, `value_vars`) MUST BE TAKEN EXACTLY FROM "Description of the DataFrame `df`" (i.e., the content of `{input_data_meta_description}`).
    2. "input_data_pickle_path": THIS MUST BE THE EXACT, UNMODIFIED STRING FROM "Path to Pickled DataFrame for Plotting" (i.e., the content of `{input_data_path}`). DO NOT CHANGE THIS PATH.
    3. "data_description_for_prompt": THIS MUST BE THE EXACT, UNMODIFIED STRING FROM "Description of the DataFrame `df`" (i.e., the content of `{input_data_meta_description}`). DO NOT CHANGE THIS DESCRIPTION.
  Example Action Input for this tool (assuming {input_data_path} is "/path/to/data.pkl" and {input_data_meta_description} specifies columns 'Actual Column A', 'Measure B'):
  `{{"task_description": "Create a pie chart. Use 'Actual Column A' for labels and 'Measure B' for values. Title: 'Chart Title'.", "input_data_pickle_path": "/path/to/data.pkl", "data_description_for_prompt": "{input_data_meta_description}"}}`

- `execute_python_code_tool`: Use this to run the Python plotting code.
  Action Input: MUST be a string which IS a valid JSON object (starts with `{{`, ends with `}}`, no outer quotes or backticks).
  This JSON object MUST have THREE keys:
    1. "python_code": The Python code string (direct output from `visualization_code_generator_tool`).
    2. "context": This MUST be the string "visualization".
    3. "input_data_pickle_path": THIS MUST BE THE EXACT, UNMODIFIED STRING FROM "Path to Pickled DataFrame for Plotting" (i.e., the content of `{input_data_path}`). DO NOT CHANGE THIS PATH. This is critical for the scaffold to load the correct df.
  Example Action Input: `{{"python_code": "import matplotlib.pyplot as plt\\n# ...plotting code...", "context": "visualization", "input_data_pickle_path": "{input_data_path}"}}`

WORKFLOW AND CRITICAL CONSIDERATIONS:
.  **ABSOLUTE DATA DESCRIPTION ADHERENCE**: Internalize the EXACT column names and types from "Description of the DataFrame `df`" ({input_data_meta_description}).
2.  **ABSOLUTE PATH ADHERENCE**: When calling `visualization_code_generator_tool` or `execute_python_code_tool`, the "input_data_pickle_path" argument you provide to these tools MUST be the exact value of `{input_data_path}`.
3.  **ALIGN TASK WITH AVAILABLE DATA**: Your plan for VCGT must use column names from {input_data_meta_description}.
4.  **FORMULATE PLOTTING PLAN**: This plan is for the "task_description" input of `visualization_code_generator_tool`. It must detail the plot type and ALL column mappings (x, y, hue, etc.), using ONLY AND EXACTLY the column names from "Description of the DataFrame `df`" ({input_data_meta_description}).
    - If `melt` is needed: Specify `id_vars` and `value_vars` using EXACT names from "Description...". Ensure `value_name` for `melt` is unique and not in "Description...".
5.  **GENERATE CODE**: Call `visualization_code_generator_tool`.
6.  **EXECUTE CODE**: Call `execute_python_code_tool` with `context="visualization"` and the correct `input_data_pickle_path`.
7.  **FINAL ANSWER**: The dictionary from `execute_python_code_tool` is your `Final Answer` JSON string.

HANDLING ERRORS from `execute_python_code_tool`:
- If `execute_python_code_tool` returns an error like "Pickle file for visualization ... not found" or "AttributeError: 'NoneType' object has no attribute ...":
    1. `Thought`: This means the `input_data_pickle_path` I provided to `execute_python_code_tool` was WRONG or the file does not exist. I MUST re-check the value of `{input_data_path}` provided to me by the orchestrator. It is MY responsibility to use the correct path.
    2. `Thought`: I will NOT assume the path is correct and retry with the same wrong path.
    3. `Thought`: If I am certain I am using the exact `{input_data_path}` and it still fails, I should report the error "The specified pickle file for visualization at '{input_data_path}' could not be found or is invalid."
    4. `Final Answer:` (If error persists after checking path) `{{"status": "error", "error_message": "The specified pickle file for visualization at '{input_data_path}' could not be found or loaded."}}`
- If `execute_python_code_tool` returns a `KeyError`:
    1. `Thought:` This means the Python code used a column name not in the DataFrame loaded from `{input_data_path}`. The column names I should use are defined by `{input_data_meta_description}`.
    2. `Thought:` I will re-read `{input_data_meta_description}` and my "task_description" to `visualization_code_generator_tool` to ensure I instructed it to use the correct column names.
    3. Re-attempt with a corrected "task_description" to `visualization_code_generator_tool`.
    
RESPONSE FORMAT (ReAct Style - FOLLOW EXACTLY):
Thought: [Your concise reasoning, step, plan for next action. Explicitly state which column names from {input_data_meta_description} you are using.]
Action: [EXACT tool name. NO formatting. ON ITS OWN LINE.]
Action Input: [Single string. If JSON, it IS the JSON object (e.g., `{{"key": "value"}}`). NO outer quotes/backticks. ON ITS OWN LINE.]
Observation: [System-filled.]
...
Thought: I have the result.
Final Answer: [If success, a valid JSON string: `{{"status": "success", "output_type": "image_path", "value": "/absolute/path/to/plot.png"}}`. If error, JSON: `{{"status": "error", "error_message": "Reason, mentioning if data description mismatch."}}`]

Begin!
Thought: {agent_scratchpad}
"""

va_prompt = PromptTemplate(
    template=VA_PROMPT_TEMPLATE_STR,
    # Nota i nomi delle variabili di input qui, devono corrispondere a ciò che passiamo in invoke
    input_variables=["input_task_description", "input_data_path", "input_data_meta_description", "agent_scratchpad"]
).partial(
    tools="\n".join([f"- {tool.name}: {tool.description}" for tool in va_tools_list]),
    tool_names=", ".join([t.name for t in va_tools_list])
)

### Create the ReAct Agent for VA
va_agent_runnable = create_react_agent(va_llm, va_tools_list, va_prompt)

### AgentExecutor for the VA
va_agent_executor = AgentExecutor(
    agent=va_agent_runnable,
    tools=va_tools_list,
    verbose=True,
    handle_parsing_errors="Ensure Action is a valid tool and Action Input is correctly formatted as per instructions.",
    max_iterations=8 # VA should also be fairly direct
)

### Runner function called by SDA to delegate a visualization task to VA
def run_visualization_agent(task_description: str, input_data_pickle_path: str, data_description_for_prompt: str) -> dict:
    """
    Invokes the VisualizationAgent to generate a plot.
    Returns a dictionary:
    - On success: e.g., {"status": "success", "output_type": "image_path", "value": "path/to/plot.png"}
    - On error: e.g., {"status": "error", "error_message": "description of error"}
    """
    print(f"[run_visualization_agent] Task: {task_description}, DataPath: {input_data_pickle_path}, DataDesc: {data_description_for_prompt[:100]}...")
    try:
        response_from_executor = va_agent_executor.invoke({
            "input_task_description": task_description,
            "input_data_path": input_data_pickle_path,
            "input_data_meta_description": data_description_for_prompt
        })
        
        agent_actual_output = response_from_executor.get("output")

        print(f"[run_visualization_agent] Type of VA's agent_actual_output: {type(agent_actual_output)}")
        print(f"[run_visualization_agent] Value of VA's agent_actual_output: {agent_actual_output}")

        if isinstance(agent_actual_output, dict):
            if "status" in agent_actual_output:
                print(f"[run_visualization_agent] VA returned a dictionary directly: {agent_actual_output}")
                return agent_actual_output
            else:
                print(f"[run_visualization_agent] VA returned a dict, but it's missing 'status' key: {agent_actual_output}")
                return {
                    "status": "error",
                    "error_message": "VA returned a dictionary output, but it was missing the required 'status' key.",
                    "raw_output": agent_actual_output
                }
        elif isinstance(agent_actual_output, str):
            print(f"[run_visualization_agent] VA returned a string, attempting JSON parse: '{agent_actual_output}'")
            try:
                parsed_output = json.loads(agent_actual_output)
                if isinstance(parsed_output, dict) and "status" in parsed_output:
                    print(f"[run_visualization_agent] VA output successfully parsed from JSON string to dict: {parsed_output}")
                    return parsed_output
                else:
                    print(f"[run_visualization_agent] VA output string parsed to non-dict or dict without 'status': {parsed_output}")
                    return {
                        "status": "error",
                        "error_message": "VA's string output, when parsed from JSON, was not a dictionary with a 'status' key.",
                        "raw_output_parsed": parsed_output,
                        "original_string": agent_actual_output
                    }
            except json.JSONDecodeError as e_json:
                print(f"[run_visualization_agent] JSONDecodeError: VA returned a string that was not valid JSON: '{agent_actual_output}'. Error: {e_json}")
                return {
                    "status": "error",
                    "error_message": f"VA returned a string that was not valid JSON: {agent_actual_output}. (JSONDecodeError: {e_json})"
                }
        else:
            print(f"[run_visualization_agent] VA returned an unexpected data type for its output: {type(agent_actual_output)}, Value: {agent_actual_output}")
            return {
                "status": "error",
                "error_message": "VisualizationAgent did not return a dictionary or a parsable JSON string for its final answer.",
                "raw_output_type": str(type(agent_actual_output)),
                "raw_output": str(agent_actual_output)
            }

    except Exception as e:
        tb_str = traceback.format_exc(limit=3)
        print(f"ERROR in VisualizationAgent runner: {type(e).__name__} - {e}\nTraceback (partial):\n{tb_str}")
        error_details = f"{type(e).__name__} - {str(e)}"
        return {
            "status": "error",
            "error_message": f"Unhandled exception in VisualizationAgent execution: {error_details}",
            "traceback_preview": tb_str.splitlines()
        }

if __name__ == '__main__':
    print("Testing VisualizationAgent...")
    # Per testare VA, avresti bisogno di un file .pkl fittizio
    # dummy_df_path = "/tmp/dummy_data_for_va.pkl"
    # import pandas as pd
    # pd.DataFrame({'category': ['A', 'B'], 'value': [10, 20]}).to_pickle(dummy_df_path)
    #
    # test_task = "Create a bar chart of 'value' by 'category'."
    # test_data_desc = "DataFrame with columns: category (str), value (int)"
    # result = run_visualization_agent(test_task, dummy_df_path, test_data_desc)
    # print(f"\nVA Result:\n{result}")
    # if os.path.exists(dummy_df_path): os.remove(dummy_df_path)
    # if result.get("status") == "success" and result.get("output_type") == "image_path":
    #     print(f"Image supposedly created at: {result.get('value')}")
    pass