# test_runner.py
import sys
import os
from pathlib import Path

# Add the agents directory to sys.path to allow imports
project_root = Path(r"C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project\data")
agents_path = Path(r"C:\Users\Utente\OneDrive\Desktop\DataScience\ML\multi_agent_data_project\agents")
sys.path.insert(0, project_root) # To allow `from agents import ...`

from agents.smart_data_analyst import run_orchestrator

if __name__ == "__main__":
    if 2 > 1:
        query = "Calculate the percentage distribution of access methods to the NoiPA portal among users aged 18–30 compared to those over 50, broken down by region of residence."
        print(f"Testing with Query: {query}\n")
        result = run_orchestrator(query)
        print("\n--- Orchestrator Result ---")
        print(result)
        print("--- End of Result ---")
    else:
        print("Please provide a query as a command-line argument.")

#Calculate the percentage distribution of access methods to the NoiPA portal among users aged 18–30 compared to those over 50, broken down by region of residence.

#Identify the most used payment method for each age group and generate a graph showing whether there are correlations between gender and payment method preference.

#Analyze commuting data to identify which administrations have the highest percentage of employees who travel more than 20 miles to work.

#Compare the gender distribution of staff among the five municipalities with the largest number of employees, highlighting any significant differences in representation by age group.

#Determine if there is a correlation between the method of portal access (from EntryAccessoAmministrati) and the average commuting distance (from EntryPendolarismo) for each administration.
