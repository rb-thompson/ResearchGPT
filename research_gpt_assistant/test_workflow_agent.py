from research_assistant import ResearchGPTAssistant
from research_agents import ResearchWorkflowAgent
from document_processor import DocumentProcessor
from config import Config

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    agent = ResearchWorkflowAgent(research_assistant)
    
    print("\033[1;32m=== Testing ResearchWorkflowAgent ===\033[0m")
    # Mock document
    doc_processor.documents["test_doc"] = {"chunks": ["AI applications include NLP and vision."]}
    
    # Test research session
    task = {"research_topic": "AI applications"}
    result = agent.execute_task(task)
    print(f"\033[1;33mResearch Session Result:\033[0m {result}")