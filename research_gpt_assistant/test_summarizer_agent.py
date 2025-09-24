from research_assistant import ResearchGPTAssistant
from research_agents import SummarizationAgent
from document_processor import DocumentProcessor
from config import Config

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    agent = SummarizationAgent(research_assistant)
    
    print("\033[1;32m=== Testing SummarizationAgent ===\033[0m")
    # Mock document
    doc_processor.documents["test_doc"] = {"chunks": ["This is a test document about AI."]}
    
    # Test single document
    result = agent.summarize_document("test_doc")
    print(f"\033[1;36mSingle Doc Summary:\033[0m {result}")
    
    # Test literature overview
    result = agent.create_literature_overview(["test_doc"])
    print(f"\033[1;33mLiterature Overview:\033[0m {result}")