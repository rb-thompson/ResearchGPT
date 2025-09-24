from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator
from document_processor import DocumentProcessor
from config import Config

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    orchestrator = AgentOrchestrator(research_assistant)
    
    print("\033[1;32m=== Testing AgentOrchestrator ===\033[0m")
    # Mock document
    doc_processor.documents["test_doc"] = {"chunks": ["AI is awesome."]}
    
    # Test routing
    task = {"doc_id": "test_doc"}
    result = orchestrator.route_task("summarizer", task)
    print(f"\033[1;36mSummarizer Result:\033[0m {result}")
    
    # Test workflow
    workflow = "Summarize and analyze AI trends"
    result = orchestrator.execute_complex_workflow(workflow)
    print(f"\033[1;33mWorkflow Result:\033[0m {result}")