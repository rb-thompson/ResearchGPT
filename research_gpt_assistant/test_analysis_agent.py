from research_assistant import ResearchGPTAssistant
from research_agents import AnalysisAgent
from document_processor import DocumentProcessor
from config import Config

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    agent = AnalysisAgent(research_assistant)
    
    print("\033[1;32m=== Testing AnalysisAgent ===\033[0m")
    # Mock document
    doc_processor.documents["test_doc"] = {"chunks": ["AI is advancing in NLP and vision."]}
    
    # Test analysis
    task = {"topic": "AI trends"}
    result = agent.execute_task(task)
    print(f"\033[1;33mAnalysis Result:\033[0m {result}")