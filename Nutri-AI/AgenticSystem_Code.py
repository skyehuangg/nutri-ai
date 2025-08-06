from agno.agent import Agent
from agno.models.anthropic import Claude
from dotenv import load_dotenv
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.csv_toolkit import CsvTools
from agno.team.team import Team

#Import thinking and reasoning for agents
from agno.tools.reasoning import ReasoningTools
from agno.playground import Playground

#Workflow Imports
from agno.workflow.v2 import Parallel, Step, Workflow
from agno.storage.sqlite import SqliteStorage

#Run Tests
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from typing import Optional
from agno.eval.performance import PerformanceEval

import os
load_dotenv()

#Defines agent to browse the web for information
def create_web_agent():
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=Claude(id="claude-3-5-haiku-20241022"),
        # reasoning_model=Claude(id="claude-3-7-sonnet-latest"),
        tools=[GoogleSearchTools()],
        instructions="You are responsible for finding healthy alternatives to the user's input (desired food)." +
                     "If needed, you will also search the Internet for the nutritional value of the food if the CsvTools Agent cannot find it through the given files." +
                     "Provide a links to where you got the information from.",
    )

    return web_agent

#Defines agent used to read CSV files
def create_csv_agent():
    csv_reader_agent = Agent(
        name="CsvTool Agent",
        role="Read a CSV file and comprehend the information. ",
        model=Claude(id="claude-3-5-haiku-20241022"),
        tools=[CsvTools(csvs=["food.csv"])],
        instructions="You are responsible for returning information about the nutritional value for the user's input." +
                    "If you are unable to find information based on the files given, give this task to the Web Agent.",
    )

    return csv_reader_agent

#Defines agent used to create summary for the agentic system
def create_reasoning_agent():
    reasoning_agent = Agent(
        name="Reasoning Agent",
        role="Compile the information from the previous information. Create a comprehensible summary and conclusion for future plans.",
        model=Claude(id="claude-3-5-haiku-20241022"),
        tools=[ReasoningTools(add_instructions=True)],
        instructions="You are responsible for compiling the information from the previous steps into a comprehensible summary and conclusion as the output." +
                    "Please organize the nutritional information as a chart and provide a bullet list of possible alternative healthy options."
    )

    return reasoning_agent

#Team used for testing
def create_team():
    agent_team = Team(
        mode="coordinate",
        members=[web_agent, csv_reader_agent, reasoning_agent],
        tools=[
            GoogleSearchTools(),
            CsvTools(csvs=["food.csv", "food_nutrient.csv", "nutrient.csv"])],
        model=Claude(id="claude-3-5-haiku-20241022")
    )
    return agent_team

#Declares and instantiates agents
web_agent = create_web_agent()
csv_reader_agent = create_csv_agent()
reasoning_agent = create_reasoning_agent()
agent_team = create_team()

#Create steps for workflow
web_agent_step = Step(
    agent = web_agent,
    name = "Web Search Phase"
)

csv_reader_step = Step(
    agent = csv_reader_agent,
    name = "CSV File Analysis Phase"
)

generation_step = Step(
    agent = reasoning_agent,
    name = "Response Generation Phase"
)

#Creates workflow
workflow = Workflow(
    name="Nutri-AI",
    description="An intelligent agentic AI system that responds to your input of food with its nutritional information. It provides a chart with values, an indepth summary, and future steps. Please enter a food you would like to review its nutritional information.",
    steps=[
        Parallel(csv_reader_step, web_agent_step),
        generation_step],

    #Code for storage and memory
    storage=SqliteStorage(
        table_name="nutri-ai_sessions",
        db_file="tmp/nutri-ai.db",
        mode="workflow_v2",
    )
)

#Creates the playground variable
playground_app = Playground(
    workflows=[workflow],
    name="Nutri-AI",
    description="Interactive UI for providing nutritional facts and alternative options",
    app_id="nutri-ai"
    )

#Links the code (backend) to the UI platform (frontend, Playground)
app = playground_app.get_app()
if __name__ == "__main__":
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    playground_app.serve(f"{module_name}:app", reload=True)

#Defines functions to run tests: accuracy and performance.
def run_accuracy_evaluation():
    evaluation = AccuracyEval(
        model=Claude(id="claude-3-5-haiku-20241022"),
        team=agent_team,
        input="I would like to know the nutritional value of chicken. Please give me alternative foods as well and a summary.",
        expected_output="Uses numeric evidence and tables to display the nutritional information of the food. Provides a summary and bullet list of alternative options.",
        num_iterations=1,
    )

    result: Optional[AccuracyResult] = evaluation.run(print_results=True)

    if result is None:
        print("⚠️ Evaluation failed or returned no result.")
    elif result.avg_score < 8:
        print(f"⚠️ Evaluation score too low: {result.avg_score}")
    else:
        print("✅ Evaluation passed.")

#run_accuracy_evaluation()

def run_performance_evaluation():
    simple_response_perf = PerformanceEval(
        func=create_team, num_iterations=1, warmup_runs=0
    )
    simple_response_perf.run(print_results=True, print_summary=True)

#run_performance_evaluation()