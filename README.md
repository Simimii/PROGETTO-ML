# PROJECT MACHINE LEARNING - REPLY

# 3SM’s Multi-Agent System 

## Sofia Matarante (TL), Simone Moroni, Simone Menghini 

### 1. Introduction 

This project presents a fully functional multi-agent system designed to analyze real-world public administration datasets through natural language interaction. The system allows users to ask questions in plain English and receive data-driven answers, either in textual or visual form, without writing a single line of code. 

The architecture is built around a network of intelligent agents, each responsible for a specific task in the data analysis pipeline. Unlike earlier prototypes where only the orchestrator had reasoning capabilities, in this final version all agents are autonomous and powered by their own language models. This design shift significantly improves performance, adaptability, and the system’s ability to handle vague or incomplete queries. 

The agents collaborate to turn user queries into actionable pipelines: selecting relevant datasets, processing data, generating visualizations, and returning insights. Each agent uses its own set of specialized tools to complete its task, contributing to a clear separation of responsibilities and modularity. The final result is a robust, end-to-end system that transforms raw CSV files into valuable information with minimal user effort. 

This project was developed by Team 3SM (Simone Moroni, Simone Menghini, Sofia Matarante(TL)). The system was developed using Python and integrates OpenAI's GPT-4o mini models, LangChain, and a custom interface. It was created as a final project for the Machine Learning course at LUISS University, academic year 2024/2025. 

### Section 2: Methods 

#### System Architecture and Design Choices 

Our multi-agent system was designed with modularity and specialization in mind. It is composed of three agents — the Orchestrator Agent, the Data Processing Agent (DPA), and the Visualization Agent — each with a specific role and a set of dedicated internal tools. The agents collaborate to transform a natural language query into an actionable insight or visual output using structured public administration data. 

- **The Orchestrator Agent** is the brain of the system. It receives the user’s query, performs intent detection, identifies the relevant dataset(s), and routes the task to the appropriate agent(s). It does not perform any data manipulation itself but coordinates the execution flow, handles inter-agent communication, and ensures smooth task delegation. 

- **The Data Processing Agent (DPA)** is responsible for dataset inspection and computation. It uses two main internal tools: 

1. Data Insight Tool: This module analyzes the structure of the dataset, infers column types and relevance, and builds an action plan based on the user’s query. 

2. Code Generation Tool: Based on the plan, this module generates executable Python code to perform the required operations. 

Code Execution Tool: This module safely runs the generated code and returns the result (as raw output or processed data) to the orchestrator. 

The Visualization Agent transforms results into visual representations. Like the Data Processing Agent, it leverages code generation and execution tools. It determines the most appropriate chart type (e.g., bar, pie, line) based on the user query, then creates the visualization using standard Python plotting libraries and saves the image for user display. 

Initially, only the Orchestrator Agent had a language model (Mistral, Tiny LLaMA), while the others lacked a model. These proved insufficient for reasoning and task generalization. After benchmarking, we upgraded all agents to GPT-4o Mini, which provided the right balance between computational efficiency and performance, especially in terms of interpretability and adaptability. 

This final configuration allows the system to operate in a loosely coupled but highly cooperative fashion, where each agent performs reasoning and decision-making in its own domain. 

Tools and Environment 

The project relies on a combination of language models, internal tools, and Python libraries to function. Here is a breakdown: 

Agent 

Tools Used 

Orchestrator Agent 

Intent Parser, Dataset Selector 

Data Processing Agent 

Data Insight Tool, Code Generator, Code Executor 

 

 

Core libraries used across all agents: 

Python 3.11, Jupyter Notebooks 

Pandas for data handling 

Matplotlib, Seaborn for chart generation 

OpenAI API (GPT-4o Mini) for reasoning and generation tasks 

Due to hardware limitations (e.g., lack of GPU support locally), model inference was performed via cloud-based API calls rather than self-hosted LLMs. 

Environment Reproducibility 

 

Environment Reproducibility 

Our project is contained in a main folder called Multi-Agent-Data-Project. Inside this folder, we’ve organized everything into five subfolders: 

Agents/ contains all the Python files that define the three agents: the Orchestrator, the Data Processing Agent, and the Visualization Agent. 

Charts/ is where all the visualizations generated by the system are saved. 

Data/ includes the four official NoiPA datasets used to answer user queries. 

Temp-Agent-Outputs/ temporarily stores small files produced during the interaction between agents, such as generated code or task logs. 

Alongside these folders, in the root of the project, you’ll also find: 

app.py, the main file that runs the Streamlit interface. 

test_runner.py, a quick way to test the system without launching the full UI. 

.env, where we store our OpenAI API key (required to run the agents). 

To launch the project, simply open a terminal inside the project folder and run: 

bash 

CopyEdit 

streamlit run app.py 

This will start the full web interface, allowing you to interact with the agents using natural language. 
