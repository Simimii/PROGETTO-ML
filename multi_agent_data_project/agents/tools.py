# agents/tools.py
import os
import sys
import traceback
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
import json
import shutil
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import subprocess
import json
import time
from pathlib import Path
import tempfile
import re
from pydantic import BaseModel, Field

### --- Initial Configuration & Setup ---
### Load environment variables, primarily the OpenAI API key.
### Define key directories: DATA_DIR for input CSVs, CHARTS_DIR for output plots,
### and TEMP_AGENT_OUTPUTS_DIR for intermediate data like pickled DataFrames.

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        PROJECT_ROOT = Path(__file__).resolve().parent.parent # Prova a salire di un solo livello se tools è in app/
        dotenv_path = PROJECT_ROOT / ".env"
    
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"[data_tools.py] Loaded .env from: {dotenv_path}")
    else:
        print(f"[data_tools.py] WARNING: .env file not found at {dotenv_path} or {project_root / '.env'}. API key must be in environment.")
except Exception as e:
    print(f"[data_tools.py] Error determining project root or loading .env: {e}")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Prova a prenderlo dall'ambiente se .env non ha funzionato o non è presente
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Ensure it's in .env at project root or set as an environment variable.")


### Initialize LLM for code generation tasks (used by DOCGT and VCGT).
### Using gpt-4o-mini with low temperature for more deterministic code output.

try:
    code_gen_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0, request_timeout=60)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOpenAI for code_gen_llm in data_tools.py: {e}")


### Define absolute paths for data and output directories.

_project_root_for_paths = Path(r'C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project')

DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts" # Per grafici finali (se VA li salva qui)
TEMP_AGENT_OUTPUTS_DIR = PROJECT_ROOT / "temp_agent_outputs" # NUOVA DIRECTORY
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True) # Crea la directory

print(f"[{Path(__file__).name}] DATA_DIR resolved to: {DATA_DIR}")
print(f"[{Path(__file__).name}] CHARTS_DIR resolved to: {CHARTS_DIR}")
print(f"[{Path(__file__).name}] TEMP_AGENT_OUTPUTS_DIR resolved to: {TEMP_AGENT_OUTPUTS_DIR}")


 ### --- Tool 1: Data Insights Tool ---
@tool
def data_insights_tool(dataset_name: str) -> str:
    """
    Inspects a specified CSV dataset to extract schema information (columns, dtypes, and a few unique example values).
    The input must be a single, valid dataset filename string located in the data directory (e.g., "MyDataset.csv").
    """
    print(f"[data_insights_tool] Requested insight for dataset: '{dataset_name}'")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        return "Error: Dataset name cannot be empty."
    
    cleaned_dataset_name = dataset_name.strip().strip("'\"")
    if not cleaned_dataset_name.endswith(".csv"):
         
        if re.match(r"^[a-zA-Z0-9_ -]+$", cleaned_dataset_name):
            print(f"[data_insights_tool] Input '{cleaned_dataset_name}' missing .csv, attempting to append.")
            cleaned_dataset_name += ".csv"
        else:
            return f"Error: Input '{cleaned_dataset_name}' does not look like a valid CSV filename base."


    file_path = DATA_DIR / cleaned_dataset_name

    if not file_path.is_file():
        available_files = [f.name for f in DATA_DIR.iterdir() if f.is_file() and f.name.endswith('.csv')]
        error_msg = (f"Error: File '{cleaned_dataset_name}' not found at path '{str(file_path)}'. "
                     f"Please ensure the filename is correct and the file exists in the '{DATA_DIR.name}' directory. "
                     f"Available CSV files: {available_files if available_files else 'None found in data directory.'}")
        print(f"ERROR [data_insights_tool]: {error_msg}")
        return error_msg
    try:
        df = pd.read_csv(file_path, nrows=5)
        if df.empty and os.path.getsize(file_path) > 0: # Se il file ha dimensione ma le prime 5 righe sono vuote o header errato
            return f"Dataset: {cleaned_dataset_name} might be empty or has issues with header/initial rows. nrows=5 returned empty DataFrame."
        elif df.empty:
            return f"Dataset: {cleaned_dataset_name} appears to be empty."
            
        schema_info_parts = [f"Dataset: {cleaned_dataset_name}"]
        schema_info_parts.append(f"Shape (first 5 rows): {df.shape}")
        schema_info_parts.append("Columns:")
        for col in df.columns:
            example_values = df[col].dropna().unique().tolist()
            example_str = str(example_values[:3]) + ('...' if len(example_values) > 3 else '')
            schema_info_parts.append(f"  - `{col}` (dtype: {df[col].dtype}): Examples: {example_str}")
        
        result = "\n".join(schema_info_parts)
        print(f"[data_insights_tool] Schema for '{cleaned_dataset_name}':\n{result[:300]}...") # Logga solo una parte
        return result
    except Exception as e:
        error_msg = f"Error inspecting dataset '{cleaned_dataset_name}': {type(e).__name__} - {str(e)}"
        print(f"ERROR [data_insights_tool]: {error_msg}")
        return error_msg

### --- Tool 2: Data Operation Code Generator Tool (DOCGT) ---
### Defines the prompt template for the LLM to generate Python/Pandas code for data manipulation.

DATA_OPERATION_CODE_PROMPT_TEXT = """
You are an expert Python/Pandas code generator.
Your task is to write a Python script to perform data operations as described.
The script will be executed in an environment where 'pandas' is imported as 'pd' and 'Path' from 'pathlib'.

Data Directory: The datasets are located in the directory: {data_dir_path}
To load a CSV, for example 'MyData.csv', your code MUST use:
`df = pd.read_csv(Path(r'{data_dir_path}') / 'MyData.csv', encoding='utf-8')` # <-- AGGIUNTO encoding='utf-8'
Ensure you use the exact filename provided in the task or schema, case-sensitively.
Assume all CSV files are UTF-8 encoded unless specified otherwise in the schema_info (which is unlikely).

User Task for Data Processing:
{task_description}

Schema of Relevant Dataset(s) (use this to inform your code, especially column names and dtypes):
{schema_info}

Critical Instructions for Code Generation:
# ... (le altre istruzioni rimangono invariate) ...
1.  Your output MUST be ONLY the Python code block. Do NOT include any explanatory text before or after the code block.
2.  Start your code directly with imports if any are needed beyond `pandas` and `pathlib.Path`.
3.  If the data processing results in a Pandas DataFrame, this final DataFrame MUST be assigned to a variable named `result_df`.
4.  If the task results in a single value (e.g., a number, string, list, or dictionary), assign this final value to a variable named `result_value`.
5.  Do NOT include any `print()` statements in your generated code. The execution environment handles output logging and result extraction.
6.  Do NOT include code to save `result_df` or `result_value` to a file (e.g., no `to_csv`, `to_pickle`). The execution environment handles this.
7.  Ensure all column names used match those in the `schema_info` if provided. Pay attention to case sensitivity.

Python Code:
"""

data_operation_code_prompt_template = PromptTemplate(
    input_variables=["task_description", "schema_info", "data_dir_path"],
    template=DATA_OPERATION_CODE_PROMPT_TEXT
)

### LLMChain specifically for data operation code generation.

data_operation_code_generator_chain = LLMChain(llm=code_gen_llm, prompt=data_operation_code_prompt_template, output_key="generated_code_text")

class DataOpCodeToolInput(BaseModel):
    task_description: str = Field(description="Detailed natural language description of the data processing task for code generation.")
    schema_info: str = Field(description="Schema information for the relevant dataset(s) as a string. Can be an empty string if not available, but prefer providing it from data_insights_tool.")

@tool 
def data_operation_code_generator_tool(json_input_string: str) -> str:
    """
    Generates Python/Pandas code to perform data operations.
    Input MUST be a JSON string with 'task_description' and 'schema_info' keys.
    This tool is typically used by the DataProcessingAgent.
    """
    print(f"[data_operation_code_generator_tool] Received raw JSON string: {json_input_string}")
    cleaned_json_string = json_input_string.strip()
    if cleaned_json_string.startswith("`") and cleaned_json_string.endswith("`"):
        cleaned_json_string = cleaned_json_string[1:-1]
        print(f"[data_operation_code_generator_tool] Cleaned backticks, now: {cleaned_json_string}")
    
    # ... (JSON parsing logic) ...
    try:
        input_dict = json.loads(cleaned_json_string)  # Handles potential backticks
        task_description = input_dict.get("task_description")
        schema_info = input_dict.get("schema_info")

        if not task_description: 
            return "# Error: JSON input string for code generator must contain 'task_description'."
        if not schema_info: 
            print(f"WARN [data_operation_code_generator_tool]: 'schema_info' is missing or empty in the input JSON. Code generation might be suboptimal.")
            schema_info = "Schema information not provided to code generator." # Default per evitare errori downstream nel template LLMChain

    except json.JSONDecodeError as e:
        return f"# Error: Invalid JSON input string for code generator: {e}. Input was: {cleaned_json_string}"
    except Exception as e: # Altri errori di parsing
        return f"# Error parsing JSON input for code generator: {type(e).__name__} - {str(e)}. Input was: {cleaned_json_string}"
    
    try:
        resolved_data_dir = str(DATA_DIR.resolve())
        response = data_operation_code_generator_chain.invoke({
            "task_description": task_description,
            "schema_info": schema_info if schema_info else "No schema info explicitly provided to code generator.",
            "data_dir_path": resolved_data_dir
        })
        code = response.get("generated_code_text", "").strip()
        
        if code.startswith("```python"): code = code[len("```python"):].strip()
        elif code.startswith("```"): code = code[len("```"):].strip()
        if code.endswith("```"): code = code[:-len("```")].strip()
            
        print(f"--------------------------------------------------------------------")
        print(f"[data_operation_code_generator_tool] FULL Generated code:\n{code}") # STAMPA IL CODICE INTERO
        print(f"--------------------------------------------------------------------")
        
        if not code: return "# Error: Code generation by LLM resulted in empty code."
        return code
    except Exception as e:
        error_msg = f"# Error during data operation code generation: {type(e).__name__} - {str(e)}"
        print(f"ERROR [data_operation_code_generator_tool]: {error_msg}")
        return error_msg


### --- Tool 3: Visualization Code Generator Tool (VCGT) ---
### Defines the prompt template for generating Python/Matplotlib/Seaborn code for plots.
### Emphasizes using exact column names from data_description_for_prompt.

VISUALIZATION_CODE_PROMPT_TEXT = """
You are an expert Python code generator for data visualizations using Matplotlib and Seaborn.
The input data for the plot is a Pandas DataFrame that will be ALREADY LOADED into a variable named `df` in the execution scope of your generated code.
The path from where `df` was loaded is '{input_data_pickle_path}' (this is for your contextual information only; your code should assume `df` exists and is loaded).

Description of the DataFrame `df` (schema with EXACT column names and dtypes - YOU MUST USE THESE EXACT COLUMN NAMES FROM THIS DESCRIPTION in your generated Python code when referring to DataFrame columns. This description is the source of truth for column names, even if the "User Task for Visualization" below uses slightly different phrasing for a column.):
{data_description_for_prompt}
# Example of data_description_for_prompt: "DataFrame has columns: ['Actual Sales Region', 'Total Revenue', 'Product Line ID']. Dtypes are: Actual Sales Region object, Total Revenue float64, Product Line ID category."

User Task for Visualization (this describes the desired plot. YOU MUST map the general concepts in this task to the EXACT column names provided in "Description of the DataFrame `df`" above):
{task_description}
# Example of task_description: "Create a Seaborn catplot (kind='bar') with 'sales region' on x-axis, 'revenue' on y-axis, and 'product line' as hue. Title='Sales Performance'."
# In this example, if "Description of the DataFrame `df`" listed ['Actual Sales Region', 'Total Revenue', 'Product Line ID'], your code MUST use these exact names, e.g., x='Actual Sales Region', y='Total Revenue', hue='Product Line ID'.

Information for Saving the Plot:
- You MUST save the plot to a specific directory. The absolute path to this directory is provided to you as: `{charts_dir_path}`
- You MUST use a specific filename for the plot. This filename is provided to you as: `{plot_filename}`

Critical Instructions for Visualization Code Generation:
1.  Your output MUST be ONLY the Python code block for the visualization. No explanatory text before or after the code.
2.  Import necessary libraries: `import matplotlib.pyplot as plt`, `import seaborn as sns`, `from pathlib import Path`. Assume `pandas as pd` and the DataFrame `df` (matching "Description of the DataFrame `df`") are already available.
3.  Based on the "User Task for Visualization" AND "Description of the DataFrame `df`", generate Python code.
    -   CRITICAL COLUMN USAGE: When using DataFrame columns in plotting functions (e.g., for `x`, `y`, `hue`, `col`, `id_vars` in `melt`, `value_vars` in `melt`), YOU MUST use the EXACT column names as listed in "Description of the DataFrame `df`" ({data_description_for_prompt}). Do not shorten, guess, change case, or use generic names from the "User Task for Visualization" if they differ from "Description of the DataFrame `df`". THE "Description of the DataFrame `df`" ({data_description_for_prompt}) IS THE SINGLE SOURCE OF TRUTH FOR COLUMN NAMES.
    -   MELT OPERATION: If "User Task for Visualization" requires reshaping data (e.g., converting wide columns like 'percentage_18_30', 'percentage_over_50' into long format for a 'hue' aesthetic), perform a `df.melt()`.
        - `id_vars`: Use EXACT column names from "Description..." that should remain as identifier variables.
        - `value_vars`: Use EXACT column names from "Description..." that represent the measures to be unpivoted.
        - `value_name`: Name for the new column containing unpivoted values. Choose a name that DOES NOT conflict with existing column names in `df` (as per "Description..."). Good examples: 'percentage_value', 'melted_data_value'.
4.  Include plot titles, axis labels, and legends as specified in "User Task for Visualization". Rotate x-axis labels if they are long and might overlap (`plt.xticks(rotation=45, ha='right')`). Use `plt.tight_layout()` if plot elements overlap.
5.  CONSTRUCTING THE SAVE PATH (MANDATORY STEPS - YOU MUST FOLLOW THIS EXACTLY):
     a.  The absolute path to the directory where the plot MUST be saved is provided to you as: `{charts_dir_path}`
     b.  The exact filename string you MUST use for the plot is provided to you as: `{plot_filename}`
     c.  In your Python code, you MUST create the save_path like this:
         `# === Beginnning of MANDATORY save path construction ===`
         `charts_dir_for_saving_plot = Path(r'''{charts_dir_path}''')`  # Use raw triple quotes for the directory path
         `name_of_plot_file = '{plot_filename}'`  # Use simple single quotes for the filename string
         `save_path = charts_dir_for_saving_plot / name_of_plot_file`
         `# === End of MANDATORY save path construction ===`
     d.  DO NOT define `_CHARTS_DIR_SCRIPT`. Use ONLY the `{charts_dir_path}` and `{plot_filename}` values passed into this prompt, as shown above to construct `save_path`.
6.  SAVE THE PLOT: After constructing `save_path` exactly as above, save the figure:
     `plt.savefig(save_path, bbox_inches='tight')`
7.  CLOSE PLOT: CRITICAL - After saving the plot, you MUST explicitly close all Matplotlib figures to free up memory and ensure the file is properly finalized. Add this line immediately after `plt.savefig()`:
     `plt.close('all')`
8.  PRINT FINAL OUTPUT: The VERY LAST LINE of your script MUST be:
     `print(f'PLOT_GENERATED_SUCCESSFULLY: {{str(save_path.resolve())}}')`
9.  Do NOT include `plt.show()` in your script. (Was step 8, now step 9)
10. Ensure correct Python indentation

Python Code:
"""

visualization_code_prompt_template = PromptTemplate(
    input_variables=["task_description", "data_description_for_prompt", "input_data_pickle_path", "charts_dir_path", "plot_filename"],
    template=VISUALIZATION_CODE_PROMPT_TEXT
)

### LLMChain specifically for visualization code generation.
visualization_code_generator_chain = LLMChain(llm=code_gen_llm, prompt=visualization_code_prompt_template, output_key="generated_code_text")

class VisCodeGenToolInput(BaseModel):
    task_description: str = Field(description="Natural language description of the visualization to generate.")
    input_data_pickle_path: str = Field(description="Absolute path to the pickled Pandas DataFrame that will be loaded as 'df' for the plot.")
    data_description_for_prompt: str = Field(description="Textual description of the DataFrame's structure (columns, dtypes) to guide the LLM.")

@tool 
def visualization_code_generator_tool(json_input_string: str) -> str:
    """
    Generates Python code (Matplotlib/Seaborn) for data visualization.
    Input MUST be a JSON string with 'task_description', 'input_data_pickle_path',
    and 'data_description_for_prompt' keys. Used by VisualizationAgent.
    """
    print(f"[visualization_code_generator_tool] Received raw JSON string: {json_input_string}")
    try:
        input_dict = json.loads(json_input_string)
        task_description = input_dict.get("task_description")
        input_data_pickle_path = input_dict.get("input_data_pickle_path")
        data_description_for_prompt = input_dict.get("data_description_for_prompt")

        if not task_description:
            return "# Error: JSON input for viz code gen must contain 'task_description'."
        if not input_data_pickle_path:
            return "# Error: JSON input for viz code gen must contain 'input_data_pickle_path'."
        if not data_description_for_prompt:
            print(f"WARN [visualization_code_generator_tool]: 'data_description_for_prompt' is missing or empty. Plot quality might be affected.")
            data_description_for_prompt = "No specific data description provided for the DataFrame 'df'. Assume common structures or use column names directly if plotting."

    except json.JSONDecodeError as e:
        return f"# Error: Invalid JSON input string for viz code gen: {e}. Input was: {json_input_string}"
    except KeyError as e: # Se get() non viene usato e una chiave manca
        return f"# Error: Missing key in JSON input for viz code gen: {e}. Input was: {json_input_string}"
    except Exception as e: # Altri errori di parsing o imprevisti
        return f"# Error processing JSON input for viz code gen: {type(e).__name__} - {str(e)}. Input was: {json_input_string}"

    print(f"[visualization_code_generator_tool] Parsed - Task: '{task_description[:100]}...', DataPath: '{input_data_pickle_path}'")
        
    # Genera un nome di file univoco per il plot
    # Usare stem dal pickle path aiuta a correlare il grafico ai dati da cui è generato
    plot_filename = f"plot_{str(int(time.time() * 1000))}_{Path(input_data_pickle_path).stem}.png"

    try:
        # Assicurati che CHARTS_DIR sia un oggetto Path e sia risolto
        resolved_charts_dir = str(CHARTS_DIR.resolve())

        response = visualization_code_generator_chain.invoke({
            "task_description": task_description,
            "input_data_pickle_path": input_data_pickle_path, # Usato nel prompt per contesto
            "data_description_for_prompt": data_description_for_prompt,
            "charts_dir_path": resolved_charts_dir, # Path assoluto per salvare
            "plot_filename": plot_filename # Nome del file da usare nello script
        })
        code = response.get("generated_code_text", "").strip()
        
        # Pulizia del codice da eventuali ```python ``` o ```
        if code.startswith("```python"): code = code[len("```python"):].strip()
        elif code.startswith("```"): code = code[len("```"):].strip()
        if code.endswith("```"): code = code[:-len("```")].strip()
            
        print(f"[visualization_code_generator_tool] Generated code (first 150 chars): {code[:150]}...") # Log più codice
        if not code: return "# Error: Visualization code generation by LLM resulted in empty code."
        return code
    except Exception as e_inner: # Errori durante la chiamata LLMChain
        error_msg = f"# Error during visualization code generation (LLMChain call): {type(e_inner).__name__} - {str(e_inner)}"
        print(f"ERROR [visualization_code_generator_tool]: {error_msg}")
        return error_msg

### --- Tool 4: Python Code Execution Tool ---
### This is a critical tool that runs the Python code generated by DOCGT or VCGT
### in a controlled subprocess. It handles loading data (for VA) and saving results.

class ExecuteCodeInput(BaseModel):
    python_code: str = Field(description="The Python code string to execute.")
    context: str = Field(default="data_processing", description="Context of execution: 'data_processing' or 'visualization'.")
    # Aggiungiamo input_data_pickle_path qui perché serve all'esecutore per caricare 'df' nel contesto di visualizzazione
    input_data_pickle_path: str = Field(default="", description="Path to pickled DataFrame, used only if context is 'visualization' to load 'df'.")


@tool
def execute_python_code_tool(json_input_string: str) -> dict:
    """
    Executes Python code in a controlled sandbox.
    Input MUST be a JSON string with keys:
      'python_code' (str): The Python code to execute.
      'context' (str): Must be 'data_processing' or 'visualization'.
      'input_data_pickle_path' (str, optional): Absolute path to a pickled Pandas DataFrame.
                                                 Required if context is 'visualization' (loads data into 'df').
                                                 Ignored if context is 'data_processing'.
    Returns a dictionary with execution 'status' and 'output_type'/'value' or 'error' message.
    """
    print(f"[execute_python_code_tool] Received raw JSON string: {json_input_string}")
    python_code: str | None = None
    context: str | None = None
    input_data_pickle_path: str = "" 

    try:
        input_dict = json.loads(json_input_string)
        python_code = input_dict.get("python_code")
        context = input_dict.get("context")
        input_data_pickle_path = input_dict.get("input_data_pickle_path", "") 

        if not python_code: 
            return {"status": "error", "error": "JSON input for execute_code must contain 'python_code' key."}
        if not context: 
            return {"status": "error", "error": "JSON input for execute_code must contain 'context' key."}

    except json.JSONDecodeError as e:
        return {"status": "error", "error": f"Invalid JSON input string for execute_code: {e}. Input was: {json_input_string}"}
    except KeyError as e: # Se get() non viene usato e una chiave obbligatoria manca
        return {"status": "error", "error": f"Missing key in JSON input for execute_code: {e}. Input was: {json_input_string}"}
    except Exception as e: # Altri errori di parsing
        return {"status": "error", "error": f"Error processing JSON input for execute_code: {type(e).__name__} - {str(e)}. Input was: {json_input_string}"}

    print(f"[execute_python_code_tool] Parsed - Context: {context}, Code (first 100): '{python_code[:100]}...'")
    if input_data_pickle_path:
        print(f"[execute_python_code_tool] Parsed - Input Data Pickle Path: {input_data_pickle_path}")


    if not python_code.strip(): # Controlla dopo il parsing
        return {"status": "error", "error": "Parsed 'python_code' is empty."}
    if context not in ["data_processing", "visualization"]:
        return {"status": "error", "error": f"Invalid 'context' parsed: '{context}'. Must be 'data_processing' or 'visualization'."}
    if context == "visualization" and not input_data_pickle_path:
        # Questo controllo è ancora valido, l'agente VA DEVE fornire il path
        return {"status": "error", "error": "For 'visualization' context, 'input_data_pickle_path' is required but was not provided in the JSON by the agent."}

    # Inizializzazioni
    exec_temp_dir_for_script_only: Path | None = None # Directory temporanea solo per il file .py
    
    try:
        ### Create a temporary directory for the script to run in.
        exec_temp_dir_for_script_only = Path(tempfile.mkdtemp(prefix="langchain_script_exec_"))
        script_path = exec_temp_dir_for_script_only / "exec_script.py"

        # Path per i file di output dei dati (pickle o json value) andranno in TEMP_AGENT_OUTPUTS_DIR
        timestamp_ms = str(int(time.time() * 1000))
        pickle_output_filename = f"result_df_{timestamp_ms}.pkl"
        json_value_output_filename = f"result_value_{timestamp_ms}.json"
        
        # Path assoluti per i file di output dei dati che lo script Python scriverà
        pickle_path_for_script_to_write = TEMP_AGENT_OUTPUTS_DIR / pickle_output_filename
        json_value_path_for_script_to_write = TEMP_AGENT_OUTPUTS_DIR / json_value_output_filename

        # Rendi questi path "safe" per l'inserimento nelle f-string dello scaffold Python
        safe_pickle_path_for_script_str = str(pickle_path_for_script_to_write.resolve()).replace('\\', '/')
        safe_json_value_path_for_script_str = str(json_value_path_for_script_to_write.resolve()).replace('\\', '/')
        
        # Path che lo script generato potrebbe usare per leggere/scrivere
        safe_resolved_data_dir_for_script = str(DATA_DIR.resolve()).replace('\\', '/')
        safe_resolved_charts_dir_for_script = str(CHARTS_DIR.resolve()).replace('\\', '/') # Per i grafici
        
        # Rendi safe anche input_data_pickle_path per l'inserimento nello scaffold
        safe_input_data_pickle_path_for_scaffold = str(Path(input_data_pickle_path)).replace('\\', '/') if input_data_pickle_path else ""


        ### Scaffold prefix: sets up imports, paths, and loads 'df' for visualization context.
        ### Includes debug prints for loaded DataFrame columns and head.
        scaffold_code_prefix = f"""
import pandas as pd
from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use('Agg') # Usa backend non interattivo per script
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', category=FutureWarning) # Spesso da Pandas/Seaborn

_DATA_DIR_SCRIPT = Path(r'''{safe_resolved_data_dir_for_script}''')
_CHARTS_DIR_SCRIPT = Path(r'''{safe_resolved_charts_dir_for_script}''')

df = None # Inizializza df
if "{context}" == "visualization":
    actual_pickle_path_str_for_script = r'''{safe_input_data_pickle_path_for_scaffold}'''
    
    print(f"EXEC_DEBUG_SCAFFOLD: Context is visualization. Attempting to load df from: {{actual_pickle_path_str_for_script}}", file=sys.stdout)

    if actual_pickle_path_str_for_script: 
        actual_pickle_path = Path(actual_pickle_path_str_for_script)
        if actual_pickle_path.is_file():
            try:
                df = pd.read_pickle(actual_pickle_path)
                print(f"EXEC_INFO: Loaded DataFrame for visualization from {{actual_pickle_path}}. Shape: {{df.shape if df is not None and hasattr(df, 'shape') else 'None or not a DataFrame'}}", file=sys.stdout)
            except Exception as e_load_pickle_viz:
                print(f"EXEC_ERROR: Failed to load DataFrame for visualization from {{actual_pickle_path}}: {{e_load_pickle_viz}}", file=sys.stderr)
                df = None 
        else:
            print(f"EXEC_ERROR: Pickle file for visualization (from tool input at '{{actual_pickle_path_str_for_script}}') not found.", file=sys.stderr)
            df = None
    else:
        print(f"EXEC_WARN: No input_data_pickle_path provided to execute_python_code_tool for visualization context. 'df' will be None unless loaded by the agent's custom code.", file=sys.stdout)
        df = None 
"""
        
        ### Scaffold suffix for data processing: handles saving result_df or result_value.
        scaffold_code_suffix_data = f"""
# ---- Output handling for data_processing context ----
_AGENT_OUTPUT_TYPE_TAG_ = "__AGENT_Формат вывода__"
_AGENT_OUTPUT_PATH_TAG_ = "__AGENT_Путь к файлу__"

if 'result_df' in locals() and isinstance(result_df, pd.DataFrame):
    try:
        print(f"EXEC_DEBUG: Attempting to pickle result_df. Shape: {{result_df.shape}}", file=sys.stdout)
        print(f"EXEC_DEBUG: result_df dtypes:\\n{{result_df.dtypes}}", file=sys.stdout)
        print(f"EXEC_DEBUG: result_df columns: {{result_df.columns.tolist()}}", file=sys.stdout)
        for col in result_df.select_dtypes(include=['object']).columns:
            try:
                examples = result_df[col].dropna().unique()[:2]
                safe_examples = [str(ex).encode('utf-8', 'replace').decode('utf-8', 'replace') for ex in examples]
                print(f"EXEC_DEBUG: Examples from object column '{{col}}': {{safe_examples}}", file=sys.stdout)
            except Exception as e_debug_print:
                print(f"EXEC_DEBUG: Error printing examples for column '{{col}}': {{e_debug_print}}", file=sys.stdout)
        
        # Salva il pickle nel path persistente e "safe"
        with open(r'''{safe_pickle_path_for_script_str}''', 'wb') as f_pickle:
             result_df.to_pickle(f_pickle)
        print(f"{{_AGENT_OUTPUT_TYPE_TAG_}}:dataframe_pickle_path")
        # Stampa il path assoluto "safe" (con forward slash)
        print(f"{{_AGENT_OUTPUT_PATH_TAG_}}:{{str(Path(r'''{safe_pickle_path_for_script_str}''').resolve())}}")
        print(f"__AGENT_RESULT_DF_COLUMNS__:{{result_df.columns.tolist()}}")
        print(f"__AGENT_RESULT_DF_DTYPES__:\\n{{result_df.dtypes.to_string()}}")
    except Exception as e_pickle:
        print(f"EXEC_ERROR: Failed to pickle result_df: {{e_pickle}}", file=sys.stderr)
elif 'result_value' in locals():
    processed_result_value = result_value 
    if hasattr(result_value, 'tolist'): processed_result_value = result_value.tolist()
    elif isinstance(result_value, (pd.Timestamp, pd.Timedelta)): processed_result_value = str(result_value)
    elif isinstance(result_value, (set)): processed_result_value = list(result_value)
    import numpy as np 
    if isinstance(processed_result_value, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        processed_result_value = int(processed_result_value)
    elif isinstance(processed_result_value, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        processed_result_value = float(processed_result_value)
    elif isinstance(processed_result_value, np.bool_):
        processed_result_value = bool(processed_result_value)
    try:
        with open(r'''{safe_json_value_path_for_script_str}''', 'w', encoding='utf-8') as f_val:
            json.dump(processed_result_value, f_val, indent=2, ensure_ascii=False)
        print(f"{{_AGENT_OUTPUT_TYPE_TAG_}}:value_json")
        print(f"{{_AGENT_OUTPUT_PATH_TAG_}}:{{str(Path(r'''{safe_json_value_path_for_script_str}''').resolve())}}")
    except TypeError as e_json_type:
        print(f"EXEC_ERROR: Failed to save result_value as JSON due to TypeError: {{e_json_type}}. Value type: {{type(result_value)}}, Value (partial): {{str(result_value)[:200]}}", file=sys.stderr)
    except Exception as e_json_dump:
        print(f"EXEC_ERROR: Failed to save result_value as JSON: {{e_json_dump}}. Value type: {{type(result_value)}}, Value (partial): {{str(result_value)[:200]}}", file=sys.stderr)
else:
    print(f"EXEC_INFO: No 'result_df' (DataFrame) or 'result_value' defined by the script.", file=sys.stdout)
"""
        
        ### Scaffold suffix for visualization: indicates plot path is printed by agent's code.
        scaffold_code_suffix_viz = f"""
# ---- Visualization code should have printed the plot path to stdout ----
# The LLM-generated code is expected to save the plot to a file (e.g., in _CHARTS_DIR_SCRIPT)
# and then print the absolute path of that saved file as its VERY LAST line of output.
# Example: print(str(Path(_CHARTS_DIR_SCRIPT / "my_plot.png").resolve()))
"""
        # Costruzione di full_code
        full_code = scaffold_code_prefix + "\n" + python_code + "\n"
        if context == "data_processing":
            full_code += scaffold_code_suffix_data
        elif context == "visualization":
            full_code += scaffold_code_suffix_viz
        
        with open(script_path, "w", encoding='utf-8') as f:
            f.write(full_code)
        print(f"[execute_python_code_tool] Script for execution ({context}) saved to: {script_path}")

        exec_env = os.environ.copy()
        exec_env["PYTHONIOENCODING"] = "utf-8"
        exec_env["PYTHONUTF8"] = "1"

        process = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace', # errors='ignore' o 'backslashreplace' se 'replace' da problemi
            cwd=exec_temp_dir_for_script_only, env=exec_env
        )
        stdout, stderr = process.stdout.strip(), process.stderr.strip()

        # Stampa output per debug
        print(f"DEBUG [execute_python_code_tool] Raw STDOUT:\n{stdout if stdout else '<empty>'}")
        if stderr:
            print(f"DEBUG [execute_python_code_tool] Raw STDERR:\n{stderr}")

        
        ### Process results based on context (data_processing or visualization).
        if process.returncode == 0:
            if context == "data_processing":
                type_match = re.search(r"__AGENT_Формат вывода__:\s*(\S+)", stdout)
                path_match = re.search(r"__AGENT_Путь к файлу__:\s*(\S+)", stdout)
                if type_match and path_match:
                    res_type, res_path_str = type_match.group(1), path_match.group(1)
                    # res_path_str dovrebbe già essere un path "safe" con forward slash
                    res_path = Path(res_path_str) # Non serve .resolve() qui se è già assoluto
                    if res_path.is_file():
                        result_dict = {}
                         # Controlla se il file esiste effettivamente
                        if res_type == "dataframe_pickle_path":
                           result_dict = {"status": "success", "output_type": "dataframe_pickle_path", "value": str(res_path)}
                        elif res_type == "value_json":
                            try:
                                with open(res_path, 'r', encoding='utf-8') as f_val:
                                    loaded_val = json.load(f_val)
                                return {"status": "success", "output_type": "value", "value": loaded_val}
                            except Exception as e_load:
                                return {"status": "error", "error": f"Failed to load JSON result from {res_path}: {e_load}", "stdout": stdout, "stderr": stderr}
                        
                        # Estrai anche lo schema del result_df se presente
                        cols_match = re.search(r"__AGENT_RESULT_DF_COLUMNS__:(.+)", stdout)
                        dtypes_match = re.search(r"__AGENT_RESULT_DF_DTYPES__:\s*\n(.+)", stdout, re.DOTALL)
                        if cols_match:
                            try:
                                result_dict["result_df_columns"] = json.loads(cols_match.group(1).replace("'", "\"")) # Converte lista stringata in lista
                            except: # Fallback se il parsing della lista fallisce
                                result_dict["result_df_columns"] = cols_match.group(1) 
                        if dtypes_match:
                            result_dict["result_df_dtypes"] = dtypes_match.group(1)
                        return result_dict
                    else: 
                        return {"status": "error", "error": f"Output file '{res_path_str}' for type '{res_type}' was declared by script but not found on disk.", "stdout": stdout, "stderr": stderr}
                elif "EXEC_INFO: No 'result_df'" in stdout:
                     return {"status": "success", "output_type": "no_output_variable", "value": "Code executed, but no 'result_df' or 'result_value' variable was defined."}
                elif stderr and "EXEC_ERROR" in stderr:
                     return {"status": "error", "error": "Error during script's output handling (e.g., pickling/JSON). Check STDERR.", "stdout": stdout, "stderr": stderr}
                else: 
                    return {"status": "error", "error": "Result tags for data_processing not found in STDOUT, but script executed without Python errors.", "stdout": stdout, "stderr": stderr}
            
            elif context == "visualization":
                # Cerca il tag specifico e il path .png
                plot_path_match = re.search(r"PLOT_GENERATED_SUCCESSFULLY:\s*(.+?\.png)", stdout, re.IGNORECASE) # Aggiunto IGNORECASE per robustezza
                
                if plot_path_match:

                    p_str_candidate = plot_path_match.group(1).strip()
                    # Il path stampato dallo script generato dovrebbe già essere assoluto e corretto,
                    # perché il prompt del generatore di codice gli dice di usare {charts_dir_path} (che è assoluto)
                    # e {plot_filename}, e poi fare .resolve().
                    potential_path = Path(p_str_candidate)
                    
                    print(f"!!!!!! [EXECUTE_TOOL_DEBUG] Attempting to check file: '{str(potential_path.resolve())}' !!!!!!!") 
                    print(f"[execute_python_code_tool] DEBUG: Matched plot path from tag: '{p_str_candidate}'")
                    print(f"[execute_python_code_tool] DEBUG: PotentialPath object: {potential_path}, is_absolute: {potential_path.is_absolute()}")

                    if potential_path.is_file(): # Verifica se il file esiste nel path stampato
                        print(f"SUCCESS [execute_tool_VIS_CONTEXT] File {potential_path.name} CONFIRMED to exist at path IMMEDIATELY after script execution.")
                        # Verifica aggiuntiva opzionale ma buona: che sia dentro la directory dei grafici attesa
                        # Questo previene se l'LLM salva in un posto completamente inaspettato ma valido.
                        expected_charts_dir_resolved = CHARTS_DIR.resolve()
                        if str(potential_path.resolve()).startswith(str(expected_charts_dir_resolved)):
                            print(f"[execute_python_code_tool] SUCCESS: Plot image found via TAG at: {potential_path.resolve()}")
                            return {"status": "success", "output_type": "image_path", "value": str(potential_path.resolve())}
                        else:
                            print(f"[execute_python_code_tool] WARNING: Plot saved at '{potential_path.resolve()}' which is outside expected charts dir '{expected_charts_dir_resolved}'. Accepting it as it exists.")
                            return {"status": "success", "output_type": "image_path", "value": str(potential_path.resolve())} # Lo accetta comunque se il file esiste
                    else: # Tag trovato, ma il path stampato non punta a un file esistente
                        return {"status": "error", "error": f"Tag PLOT_GENERATED_SUCCESSFULLY found, but path '{p_str_candidate}' does not point to an existing file.", "stdout": stdout, "stderr": stderr}
                else: # Tag PLOT_GENERATED_SUCCESSFULLY non trovato
                    # Qui puoi mettere il fallback se vuoi provare a cercare path .png generici,
                    # ma se il tag non c'è, è probabile che l'LLM non abbia seguito le istruzioni.
                    print(f"[execute_python_code_tool] ERROR: PLOT_GENERATED_SUCCESSFULLY tag NOT found in STDOUT.")
                    # Controlla se c'è un path .png qualsiasi nello stdout come ultima risorsa
                    fallback_candidates = re.findall(r"([a-zA-Z]:[\\/].+?\.png|/[^'\"]+?\.png|[\w\-\./\\]+?\.png)", stdout)
                    if fallback_candidates:
                        last_png_path = Path(fallback_candidates[-1].strip())
                        if last_png_path.is_file():
                             print(f"[execute_python_code_tool] WARNING: Fallback found a .png path without tag: {last_png_path}")
                             return {"status": "success", "output_type": "image_path", "value": str(last_png_path.resolve())}

                    
                    return {"status": "error", "error": "PLOT_GENERATED_SUCCESSFULLY tag not found in STDOUT for visualization, and no fallback .png path found.", "stdout": stdout, "stderr": stderr}
        else: # process.returncode != 0
            error_detail = stderr if stderr else stdout # Python errors spesso vanno su stderr
            return {"status": "error", "error": f"Code execution failed with return code {process.returncode}. Details: {error_detail}", "stdout": stdout, "stderr": stderr}

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Execution timed out after 120 seconds. Check the code for infinite loops or long-running operations."}
    except Exception as e_outer: # Altri errori nell'impalcatura del tool stesso
        tb_str = traceback.format_exc(limit=2)
        print(f"Outer ExecuteCodeTool error: {type(e_outer).__name__} - {str(e_outer)}\n{tb_str}")
        return {"status": "error", "error": f"Outer ExecuteCodeTool error: {type(e_outer).__name__} - {str(e_outer)}"}
    finally:
        if exec_temp_dir_for_script_only and exec_temp_dir_for_script_only.exists():
            try:
                shutil.rmtree(exec_temp_dir_for_script_only)
                print(f"[execute_python_code_tool] Cleaned up script temp dir: {exec_temp_dir_for_script_only}")
            except Exception as e_clean:
                print(f"WARN [execute_python_code_tool] Failed to clean up script temp dir {exec_temp_dir_for_script_only}: {e_clean}")


### --- Tool 5: Format DataFrame Tool ---
@tool
def format_dataframe_tool(pickle_path: str, max_rows_to_display: int = 10) -> str:
    """
    Loads a Pandas DataFrame from a pickle file and returns a Markdown string representation
    of its head (up to max_rows_to_display) and its columns.
    If the DataFrame has fewer rows than max_rows_to_display, it displays all rows.
    Input:
        pickle_path (str): The absolute path to the .pkl file.
        max_rows_to_display (int): Maximum number of rows to include in the Markdown table.
    """
    try:
        df = pd.read_pickle(pickle_path)
        num_rows = len(df)
        num_cols = len(df.columns)
        cols_str = ", ".join(df.columns.tolist())

        if num_rows == 0:
            return f"The DataFrame at {pickle_path} is empty. Columns: [{cols_str}]."

        display_df = df.head(max_rows_to_display)
        markdown_table = tabulate(display_df, headers='keys', tablefmt='pipe') # 'pipe' è il formato Markdown

        if num_rows > max_rows_to_display:
            summary = (
                f"Showing the first {max_rows_to_display} of {num_rows} rows from the DataFrame "
                f"(columns: [{cols_str}]):\n\n{markdown_table}\n\n"
                f"The complete dataset is saved at: {pickle_path}"
            )
        else:
            summary = (
                f"The DataFrame (columns: [{cols_str}], {num_rows} row(s)) is:\n\n{markdown_table}\n\n"
                f"The dataset is also saved at: {pickle_path}"
            )
        return summary
    except FileNotFoundError:
        return f"Error: Pickle file not found at {pickle_path}."
    except Exception as e:
        return f"Error loading or formatting DataFrame from {pickle_path}: {str(e)}"
   
    
### --- Tool 6: Insight Explanation Tool ---
### LLM and chain setup specifically for generating explanations.

try:
    # Usiamo un LLM capace per l'interpretazione, GPT-4o-mini va bene.
    # Potresti considerare una temperatura leggermente più alta per spiegazioni più "creative".
    explanation_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOpenAI for explanation_llm: {e}")

EXPLANATION_PROMPT_TEMPLATE = """
You are an expert data analyst tasked with explaining insights from data or visualizations.
You will be given:
1.  The original user query that led to this data/visualization.
2.  A description of the data that was processed or visualized (e.g., column names, types).
3.  The data itself (either as a Markdown table or a description of a chart's content and its path).

Your goal is to provide a concise, easy-to-understand explanation of the key insights derived from the data/visualization in relation to the user's original query.
Focus on what the data *means* in the context of the query. Avoid simply re-stating the data.
If it's a chart, explain what the chart shows and what conclusions can be drawn.
If it's tabular data, highlight significant patterns, totals, or comparisons.
Keep the explanation to 2-4 sentences if possible.

Original User Query:
{user_query}

Description of Processed/Visualized Data:
{data_description}

Data/Visualization Content:
{data_content_or_chart_description}
{chart_path_mention_if_any}

Based on all the above, provide your insightful explanation:
"""

explanation_prompt = PromptTemplate(
    template=EXPLANATION_PROMPT_TEMPLATE,
    input_variables=["user_query", "data_description", "data_content_or_chart_description", "chart_path_mention_if_any"]
)

explanation_chain = LLMChain(llm=explanation_llm, prompt=explanation_prompt)

class ExplanationToolInput(BaseModel):
    user_query: str = Field(description="The original query from the user that led to the result being explained.")
    data_description: str = Field(description="A description of the data that was processed or visualized (e.g., 'DataFrame with columns X, Y. X is category, Y is value').")
    data_content_or_chart_description: str = Field(description="Either the actual data (e.g., Markdown table) or a description of what a chart shows (e.g., 'A bar chart of sales by product').")
    chart_path: str = Field(default="", description="Optional: The path to the generated chart file, if the explanation is about a chart. Can be an empty string.")


@tool
def generate_insight_explanation_tool(json_input_string: str) -> str:
    """
    Generates a brief, insightful explanation for a given piece of data or a chart,
    considering the original user query.
    Input MUST be a JSON string with keys: 'user_query', 'data_description',
    'data_content_or_chart_description', and optionally 'chart_path'.
    """
    print(f"[generate_insight_explanation_tool] Received raw JSON string: {json_input_string}")
    try:
        input_data = json.loads(json_input_string)
        user_query = input_data["user_query"]
        data_description = input_data["data_description"]
        data_content = input_data["data_content_or_chart_description"]
        chart_path = input_data.get("chart_path", "") # Optional

        chart_path_mention = f"The chart can be found at: {chart_path}" if chart_path else "This explanation is based on tabular data."

        response = explanation_chain.invoke({
            "user_query": user_query,
            "data_description": data_description,
            "data_content_or_chart_description": data_content,
            "chart_path_mention_if_any": chart_path_mention
        })
        explanation = response.get("text", "Could not generate an explanation.").strip()
        print(f"[generate_insight_explanation_tool] Generated explanation: {explanation}")
        return explanation
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input to explanation tool: {e}. Input was: {json_input_string}"
    except KeyError as e:
        return f"Error: Missing key in JSON input for explanation tool: {e}. Input was: {json_input_string}"
    except Exception as e:
        return f"Error in explanation tool: {type(e).__name__} - {str(e)}"
