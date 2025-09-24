from research_assistant import ResearchGPTAssistant
from research_agents import QAAgent
from document_processor import DocumentProcessor
from config import Config

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)
    research_assistant = ResearchGPTAssistant(config, doc_processor)
    agent = QAAgent(research_assistant)
    
    print("\033[1;32m=== Testing QAAgent ===\033[0m")
    # Mock document
    doc_processor.documents["test_doc"] = {"chunks": ["AI uses neural networks for learning."]}
    
    # Test factual question
    factual_task = {"question": "What is AI?", "type": "factual"}
    result = agent.execute_task(factual_task)
    print(f"\033[1;36mFactual Answer:\033[0m {result}")
    
    # Test analytical question
    analytical_task = {"question": "Why is AI effective?", "type": "analytical"}
    result = agent.execute_task(analytical_task)
    print(f"\033[1;33mAnalytical Answer:\033[0m {result}")