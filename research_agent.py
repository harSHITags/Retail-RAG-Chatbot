import os
import datetime
import traceback
from crewai import LLM
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv(override=True)

class DuckDuckGoSearchTool(BaseTool):
    name: str = "Search Web Tool"
    description: str = "Useful to search the web for information about the query."

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found for the query."
            formatted_results = []
            for r in results:
                title = r.get("title", "No Title")
                href = r.get("href", "No URL")
                body = r.get("body", "No Summary")
                formatted_results.append(f"Title: {title}\nURL: {href}\nSummary: {body}\n")
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"

class RetailResearchAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if self.model_name in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
            self.model_name = "llama-3.3-70b-versatile"
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            print("WARNING: GROQ_API_KEY is not set or is using the default placeholder.")
            
        self.llm = LLM(
            model=f"groq/{self.model_name}",
            api_key=self.api_key,
            temperature=0.3
        )
        self.output_dir = os.path.join("knowledge_repository", "research_findings")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_research_crew(self, user_query: str) -> str:
        
        search_tool = DuckDuckGoSearchTool()

        researcher = Agent(
            role='Senior Retail Research Analyst',
            goal=f'Conduct comprehensive research on the following query: {user_query}',
            backstory='You are an expert retail research analyst with a keen eye for market trends, consumer behavior, and retail operations. You excel at finding accurate and up-to-date information.',
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
            llm=self.llm
        )

        synthesizer = Agent(
            role='Retail Strategy Consultant',
            goal='Synthesize research findings into a structured, actionable report.',
            backstory='You are a seasoned retail strategy consultant known for distilling complex data into clear, concise, and highly actionable insights. You communicate in a professional, structured manner.',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        research_task = Task(
            description=f'Search the web for relevant information regarding: "{user_query}". Gather facts, concrete statistics, market data, and expert opinions. Pay special attention to numerical data that can be visualized.',
            expected_output='A detailed summary of the findings from the web search, heavily emphasizing statistical data, metrics, and quantitative trends.',
            agent=researcher
        )

        synthesis_task = Task(
            description='''Review the findings from the research task and create a highly detailed, analytical report. 
            The report MUST include the following sections, formatted in Markdown:
            - **Executive Summary**
            - **Data & Trends Analysis** (Include at least one detailed Markdown table comparing key metrics or trends)
            - **Visual Insights** (Include at least one Mermaid.js chart, such as a pie, bar, or line chart to visualize the data. Use ```mermaid code blocks)
            - **Key Findings**
            - **Recommendations**
            - **Sources**
            ''',
            expected_output='A highly structured markdown report containing Executive Summary, Data & Trends Analysis with tables, Visual Insights with Mermaid.js charts, Key Findings, Recommendations, and Sources.',
            agent=synthesizer
        )

        crew = Crew(
            agents=[researcher, synthesizer],
            tasks=[research_task, synthesis_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        return str(result)

    def run_research(self, query: str) -> dict:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join([c if c.isalnum() else "_" for c in query])[:30]
        if not safe_query:
            safe_query = "query"
        filename = f"{timestamp_str}_{safe_query}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            summary = self.create_research_crew(query)
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary)
                
            return {
                "summary": summary,
                "saved_to": filepath,
                "model_used": self.model_name,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Research process failed: {str(e)}")
