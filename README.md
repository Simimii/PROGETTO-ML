# PROJECT MACHINE LEARNING - REPLY

# 3SM’s Multi-Agent System 

## Sofia Matarante (TL), Simone Moroni, Simone Menghini 

## 1. Introduction 

This project presents a fully functional **multi-agent system** designed to analyze real-world public administration datasets through **natural language interaction**. The system allows users to ask questions in plain English and receive **data-driven** answers, either in textual or visual form, without writing a single line of **code**. 

The architecture is built around a **network** of intelligent **agents**, each responsible for a **specific task** in the data analysis pipeline. Unlike earlier prototypes where only the orchestrator had reasoning capabilities, in this final version all agents are autonomous and powered by their own **language models**. This design shift significantly improves performance, adaptability, and the **system’s ability** to handle vague or incomplete queries. 

The agents **collaborate** to turn user queries into actionable **pipelines**: selecting relevant datasets, processing data, generating visualizations, and returning insights. Each agent uses its own set of specialized tools to complete its task, contributing to a clear separation of responsibilities and modularity. The final result is a robust, **end-to-end system** that transforms raw **CSV files** into valuable information with minimal user effort. 


## Section 2: Methods 

#### System Architecture and Design Choices 

Our **multi-agent system** was designed with modularity and specialization in mind. It is composed of three agents — the Orchestrator Agent, the Data Processing Agent (DPA), and the Visualization Agent — each with a specific role and a set of dedicated **internal tools**. The agents collaborate to **transform** a natural language query into an actionable insight or visual output using **structured** public administration data. 

- **The Orchestrator Agent** is the brain of the system. It receives the user’s query, performs intent **detection**, identifies the relevant dataset(s), and routes the task to the appropriate agent(s). It does not perform any **data manipulation** itself but coordinates the execution flow, handles inter-agent **communication**, and ensures smooth task delegation.

The Orchestrator also relies on two internal tools to enhance user interaction:
  1. **Format DataFrame Tool**
     When the Data Processing Agent returns the result as a pickle file, this tool converts the output into a readable table that is displayed within the platform interface.

  2. **Generate Insight Explanation Tool**
     Once the final results are available, this tool uses a language model to produce a textual summary and explanation of the outcome — making the insights easier to understand for the 
     end user.

- **The Data Processing Agent (DPA)** is responsible for dataset inspection and computation. It uses two main internal tools: 

  1. **Data Insight Tool**: This module analyzes the structure of the dataset, infers column types and relevance, and builds an action plan based on the user’s query.

  2. **Code Generation Tool**: Based on the plan, this module generates executable **Python code** to perform the required operations.

  3. **Code Execution Tool**: This module safely runs the generated code and returns the result (as raw output or processed data) to the orchestrator.

- **The Visualization Agent** transforms results into visual representations. Like the **Data Processing Agent**, it leverages code generation and execution tools. It determines the most appropriate chart type (e.g., bar, pie, line) based on the user query, then creates the visualization using standard **Python** plotting libraries and saves the image for user display. 

Initially, only the Orchestrator Agent had a language model **(Mistral, Tiny LLaMA)**, while the others lacked a model. These proved insufficient for reasoning and task generalization. After benchmarking, we upgraded all agents to **GPT-4o Mini**, which provided the right balance between computational efficiency and **performance**, especially in terms of interpretability and **adaptability**. 


In the final version of our system, the **Orchestrator Agent** has been enhanced with two advanced features:  
- **Interactive Memory** and **Proactive Follow-up Suggestions** based on user queries and results.

After generating an answer — including charts, tables, or textual insights — the system automatically suggests **1–2 related questions** that the user might find interesting. These follow-ups are context-aware and based on the data just presented.  
For example:  
*“Would you like to compare this by age group?”*  
*“Do you want to see how this varies across regions?”*


#### Handling Interactive Refinement

The system now supports **refinements of previous results** in a conversational way.  
If the user asks to adjust a recent chart or data table, the Orchestrator:

1. **Identifies the previous result** being referred to.
2. **Retrieves memory context**, including:
   - The exact `dataframe_pickle_path`
   - The list of column names (`result_df_columns`)
   - The corresponding data types (`result_df_dtypes`)
3. **Generates a new task** for the appropriate agent:
   - For chart modifications → a new task is sent to the **Visualization Agent**
   - For table edits or new filters → a new task is sent to the **Data Processing Agent**

All refinements are applied using the **original intermediate result** stored in the temporary folder.  
This avoids repeating the full data processing pipeline and enables quick, focused adjustments.


Thanks to this memory-based approach, the system now supports **fluid, multi-turn conversations** and makes data exploration more natural, personalized, and intelligent.

### Tools and Environment 

The project relies on a combination of language models, internal tools, and Python libraries to function. 


#### Core libraries used across all agents: 

- **Python 3.11, Jupyter Notebooks** 

- **Pandas** for data handling 

- **Matplotlib, Seaborn** for chart generation 

- **OpenAI API (GPT-4o Mini**) for reasoning and generation tasks 

Due to hardware limitations (e.g., lack of GPU support locally), model inference was performed via cloud-based API calls rather than self-hosted LLMs. 

### Environment Reproducibility 

Our project is contained in a main folder called ***Multi-Agent-Data-Project***. Inside this folder, we’ve organized everything into five subfolders: 

- ***Agents*** contains all the Python files that define the three agents: the Orchestrator, the Data Processing Agent, and the Visualization Agent. 

- ***Charts*** is where all the visualizations generated by the system are saved. 

- ***Data*** includes the four official **NoiPA** datasets used to answer user queries. 

- ***Temp-Agent-Outputs*** temporarily stores small files produced during the interaction between agents, such as generated code or task logs. 

Alongside these folders, in the root of the project, you’ll also find: 

- ***app.py***, the main file that runs the Streamlit interface. 

- ***test_runner.py***, a quick way to test the system without launching the full UI. 

- ***.env***, where we store our OpenAI API key (required to run the agents). 

To launch the project, simply open a terminal inside the project folder and run: 


***streamlit run app.py*** 

This will start the full web interface, allowing you to interact with the agents using natural language. 

## Section 3 and 4 : Experimental Design
To ensure that the answers **provided** by the agent were accurate, we decided to verify them manually. For each question, we compared the agent’s response with the result we obtained by running the same query in **Python**.
We **created** a **Google Sheet** containing all the main questions, the agent's answers, our manual checks, and a column indicating whether the results matched (Yes or No).
Here is the link: https://docs.google.com/spreadsheets/d/1TME5bDN4Kpig5EBdHYvW1UYpr-MtCxNtDnarzGVJBCA/edit?gid=0#gid=0


## Section 5: Conculsion 
This project showed us how **powerful** a **multi-agent system** can be when it comes to interacting with structured data in a more human way. Starting from a basic setup that didn’t work well, we **learned** through trial and error how important it is for all agents to be **smart** and work **together**. Once we equipped each agent with a **capable** model and the right **tools**, the system became able to understand **queries**, generate code, produce visualizations, and even explain the results in plain language. It was rewarding to see how much more accessible data becomes when the interaction feels conversational.

That said, there are still things to improve. The **system** sometimes has trouble when the question is too complex. Also, while we **tested** many queries, there’s still no automatic way to tell if the answer is fully correct — we had to check that **manually**. In the future, it would be interesting to add more robust memory for **long conversations**, and maybe integrate some kind of automatic validation to better trust the **outputs**.
