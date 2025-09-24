from research_assistant import ResearchGPTAssistant
from research_agents import BaseAgent
from config import Config

# Mock agent for testing
class MockAgent(BaseAgent):
    def execute_task(self, task_input):
        prompt = f"Mock task: {task_input}"
        return self._call_mistral(prompt)

# Test BaseAgent
if __name__ == "__main__":
    config = Config()
    research_assistant = ResearchGPTAssistant(config, None)  # No doc_processor needed
    agent = MockAgent(research_assistant)
    print("\033[1;32m=== Testing BaseAgent ===\033[0m")  # Neon green
    result = agent.execute_task("Test Mistral API call")
    print(f"\033[1;36mResult: {result}\033[0m")  # Cyan