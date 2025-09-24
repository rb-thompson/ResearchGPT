from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from logging import StreamHandler, Formatter

def setup_logger(name: str) -> logging.Logger:
    """Configure a logger with retro-style ANSI color formatting.

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.handlers = []  # Prevent duplicate handlers
    handler = StreamHandler()
    handler.setFormatter(Formatter(
        "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

class BaseAgent(ABC):
    """Abstract base class for all research agents."""
    
    def __init__(self, research_assistant: Any) -> None:
        """Initialize BaseAgent with research assistant reference.

        Args:
            research_assistant: Instance of ResearchGPTAssistant for API access.
        """
        self.assistant = research_assistant
        self.agent_name: str = "BaseAgent"
        self.logger: logging.Logger = setup_logger(self.agent_name)
        self.logger.info("🟢 BaseAgent initialized! Ready to rock!")

    def _call_mistral(self, prompt: str, temperature: float | None = None) -> str:
        """Call Mistral API with the given prompt.

        Args:
            prompt (str): Input prompt for the Mistral API.
            temperature (float, optional): Sampling temperature. Defaults to config value.

        Returns:
            str: API response or error message.
        """
        if temperature is None:
            temperature = self.assistant.config.TEMPERATURE
        self.logger.info("🚀 Sending prompt to Mistral API...")
        try:
            response = self.assistant.mistral_client.chat.complete(
                model=self.assistant.config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.assistant.config.MAX_TOKENS
            )
            self.logger.info("🎉 Mistral responded back!")
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"❌ Mistral API error: {str(e)}")
            return f"Error: {str(e)}"

    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task.

        Args:
            task_input: Task parameters.

        Returns:
            Dict containing task results or error message.
        """
        raise NotImplementedError("Each agent must implement execute_task method")

class SummarizationAgent(BaseAgent):
    """Agent for document summarization tasks."""
    
    def __init__(self, research_assistant: Any) -> None:
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("📝 SummarizerAgent ready to summarize!")

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """Summarize a specific document.

        Args:
            doc_id (str): Document identifier.

        Returns:
            Dict containing summary, word count, and key topics or error message.
        """
        self.logger.info(f"📄 Summarizing document: {doc_id}")
        try:
            document_text = "\n".join(self.assistant.doc_processor.documents.get(doc_id, {}).get("chunks", []))
            if not document_text:
                self.logger.error(f"❌ No text found for doc_id: {doc_id}")
                return {"error": f"No text for {doc_id}"}
            
            summary_prompt = f"""
            Summarize the following document in 100-150 words:
            - Extract main research question/hypothesis
            - Describe methodology
            - List key findings
            - Note conclusions and limitations
            Document: {document_text}
            """
            summary = self._call_mistral(summary_prompt)
            
            topic_prompt = f"Extract 3-5 key topics from: {summary}"
            topics = self._call_mistral(topic_prompt).split("\n")
            
            summary_data = {
                "doc_id": doc_id,
                "summary": summary,
                "word_count": len(summary.split()),
                "key_topics": [t.strip() for t in topics if t.strip()]
            }
            self.logger.info(f"🎉 Summary generated! Word count: {summary_data['word_count']}")
            return summary_data
        except Exception as e:
            self.logger.error(f"❌ Error summarizing: {str(e)}")
            return {"error": str(e)}

    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Create a literature overview from multiple documents.

        Args:
            doc_ids (List[str]): List of document identifiers.

        Returns:
            Dict containing overview, number of papers analyzed, and individual summaries.
        """
        self.logger.info(f"📚 Creating literature overview for {len(doc_ids)} documents")
        individual_summaries = []
        for doc_id in doc_ids:
            summary = self.summarize_document(doc_id)
            individual_summaries.append(summary)
        
        summaries_text = "\n".join([s.get("summary", "") for s in individual_summaries if "summary" in s])
        overview_prompt = f"""
        Synthesize a literature overview from these summaries:
        - Identify common research themes
        - Note different methodologies
        - Highlight consistent findings vs contradictions
        - Suggest research gaps
        Summaries: {summaries_text}
        """
        overview = self._call_mistral(overview_prompt)
        
        result = {
            "overview": overview,
            "papers_analyzed": len(doc_ids),
            "individual_summaries": individual_summaries
        }
        self.logger.info("🌟 Literature overview completed!")
        return result

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        else:
            self.logger.error("❌ Invalid task input")
            return {"error": "Invalid task input for SummarizerAgent"}

class QAAgent(BaseAgent):
    """Agent for question-answering tasks."""
    
    def __init__(self, research_assistant: Any) -> None:
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("❓ QAAgent ready to answer questions!")

    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        """Answer factual questions based on document corpus.

        Args:
            question (str): Factual question to answer.

        Returns:
            Dict containing answer, sources, and confidence score.
        """
        self.logger.info(f"❓ Answering factual question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=3)
            context = "\n".join([chunk[0] for chunk in relevant_chunks])
            qa_prompt = f"""
            Answer the question concisely based on the context. Cite sources and estimate confidence (0-1).
            Question: {question}
            Context: {context}
            """
            answer = self._call_mistral(qa_prompt)
            
            confidence_prompt = f"Estimate confidence (0-1) for this answer: {answer}"
            confidence = self._call_mistral(confidence_prompt)
            
            result = {
                "question": question,
                "answer": answer,
                "sources": [chunk[2] for chunk in relevant_chunks],
                "confidence": confidence
            }
            self.logger.info(f"🎉 Factual answer generated! Confidence: {confidence}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error answering: {str(e)}")
            return {"error": str(e)}

    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        """Answer analytical questions requiring reasoning.

        Args:
            question (str): Analytical question to answer.

        Returns:
            Dict containing analysis and reasoning type.
        """
        self.logger.info(f"🤔 Answering analytical question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=5)
            response = self.assistant.chain_of_thought_reasoning(question, relevant_chunks)
            result = {
                "question": question,
                "analysis": response,
                "reasoning_type": "chain_of_thought"
            }
            self.logger.info("🌟 Analytical answer generated!")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error analyzing: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        question = task_input.get("question", "")
        question_type = task_input.get("type", "factual")
        if question_type == "analytical":
            return self.answer_analytical_question(question)
        return self.answer_factual_question(question)

class AnalysisAgent(BaseAgent):
    """Agent for generating research insights and gaps."""
    
    def __init__(self, research_assistant: Any) -> None:
        super().__init__(research_assistant)
        self.agent_name = "AnalysisAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("🔍 AnalysisAgent ready to uncover insights!")

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a topic for trends, gaps, and future directions.

        Args:
            task_input: Dictionary with 'topic' key.

        Returns:
            Dict containing analysis and sources.
        """
        self.logger.info(f"🔍 Analyzing topic: {task_input.get('topic', '')}")
        try:
            topic = task_input.get("topic", "")
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(topic, top_k=5)
            context = "\n".join([chunk[0] for chunk in relevant_chunks])
            analysis_prompt = f"""
            Analyze the following context for:
            - Key trends in the research
            - Research gaps or unanswered questions
            - Suggested future directions
            Context: {context}
            """
            analysis = self._call_mistral(analysis_prompt)
            result = {
                "topic": topic,
                "analysis": analysis,
                "sources": [chunk[2] for chunk in relevant_chunks]
            }
            self.logger.info("🌟 Analysis generated!")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error analyzing: {str(e)}")
            return {"error": str(e)}

class ResearchWorkflowAgent(BaseAgent):
    """Agent for managing complete research workflows."""
    
    def __init__(self, research_assistant: Any) -> None:
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        self.logger = setup_logger(self.agent_name)
        self.summarizer = SummarizationAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)
        self.analysis_agent = AnalysisAgent(research_assistant)
        self.logger.info("🧠 ResearchWorkflowAgent ready to run research sessions!")

    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        """Conduct a complete research session on a topic.

        Args:
            research_topic (str): Topic to research.

        Returns:
            Dict containing questions, document analysis, answers, and research gaps.
        """
        self.logger.info(f"🧠 Starting research session on: {research_topic}")
        session_results = {
            "research_topic": research_topic,
            "generated_questions": [],
            "document_analysis": {},
            "answers": [],
            "research_gaps": []
        }
        try:
            # Step 1: Generate questions
            questions_prompt = f"Generate 3-5 specific research questions for: {research_topic}"
            questions = self._call_mistral(questions_prompt).split("\n")
            session_results["generated_questions"] = [q.strip() for q in questions if q.strip()]
            self.logger.info(f"❓ Generated {len(session_results['generated_questions'])} questions")

            # Step 2: Summarize documents
            relevant_docs = self.assistant.doc_processor.find_similar_chunks(research_topic, top_k=10)
            doc_ids = list(set([doc[2] for doc in relevant_docs]))
            if doc_ids:
                session_results["document_analysis"] = self.summarizer.create_literature_overview(doc_ids)
                self.logger.info("📚 Document analysis completed")

            # Step 3: Answer questions
            for question in session_results["generated_questions"]:
                qa_task = {"question": question, "type": "analytical"}
                answer = self.qa_agent.execute_task(qa_task)
                session_results["answers"].append(answer)
                self.logger.info(f"❓ Answered: {question}")

            # Step 4: Identify gaps
            analysis_task = {"topic": research_topic}
            gaps = self.analysis_agent.execute_task(analysis_task)
            session_results["research_gaps"] = gaps.get("analysis", "")
            self.logger.info("🔍 Research gaps identified")

            self.logger.info("🌟 Research session completed!")
            return session_results
        except Exception as e:
            self.logger.error(f"❌ Error in research session: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "research_topic" in task_input:
            return self.conduct_research_session(task_input["research_topic"])
        self.logger.error("❌ Invalid task input")
        return {"error": "Invalid task input for ResearchWorkflowAgent"}

class AgentOrchestrator:
    """Orchestrates multiple agents for complex research tasks."""
    
    def __init__(self, research_assistant: Any) -> None:
        """Initialize orchestrator with agent registry and shared memory.

        Args:
            research_assistant: Instance of ResearchGPTAssistant for API access.
        """
        self.assistant = research_assistant
        self.logger = setup_logger("AgentOrchestrator")
        self.agents = {
            "summarizer": SummarizationAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "analysis": AnalysisAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant)
        }
        self.shared_memory: Dict[str, Any] = {}
        self.logger.info("🎶 AgentOrchestrator ready to conduct the symphony!")

    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route tasks to appropriate agents.

        Args:
            task_type (str): Type of task ('summarizer', 'qa', 'analysis', 'workflow').
            task_input: Task parameters.

        Returns:
            Dict containing task results or error message.
        """
        self.logger.info(f"🚦 Routing task type: {task_type}")
        try:
            agent = self.agents.get(task_type)
            if not agent:
                self.logger.error(f"❌ Unknown task type: {task_type}")
                return {"error": f"Unknown task type: {task_type}"}
            result = agent.execute_task(task_input)
            self.shared_memory[task_type] = result
            self.logger.info(f"🎉 Task routed to {agent.agent_name}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error routing task: {str(e)}")
            return {"error": str(e)}

    def resolve_conflicts(self, outputs: List[Dict[str, Any]]) -> str:
        """Resolve conflicts between agent outputs.

        Args:
            outputs: List of agent outputs.

        Returns:
            str: Resolved output or error message.
        """
        self.logger.info("🛠 Resolving conflicts in agent outputs")
        try:
            conflict_prompt = f"Identify and resolve contradictions in these outputs: {outputs}"
            resolved_output = self.agents["workflow"]._call_mistral(conflict_prompt)
            self.logger.info("🌟 Conflicts resolved!")
            return resolved_output
        except Exception as e:
            self.logger.error(f"❌ Error resolving conflicts: {str(e)}")
            return f"Error: {str(e)}"

    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """Execute a complex multi-agent workflow.

        Args:
            workflow_description (str): Natural language description of the workflow.

        Returns:
            Dict containing workflow description, executed steps, and final result.
        """
        self.logger.info(f"🎬 Executing complex workflow: {workflow_description}")
        try:
            # Parse workflow description with Mistral
            parse_prompt = f"""
            Break down this workflow into unique tasks (summarizer, qa, analysis, workflow).
            Ensure each task type is included only once unless explicitly required multiple times.
            Workflow: {workflow_description}
            """
            task_list = self.agents["workflow"]._call_mistral(parse_prompt).split("\n")
            tasks = []
            seen_tasks = set()
            for task in task_list:
                task = task.strip()
                if not task:
                    continue
                if "summarize" in task.lower() and "summarizer" not in seen_tasks:
                    tasks.append(("summarizer", {"doc_ids": list(self.assistant.doc_processor.documents.keys())}))
                    seen_tasks.add("summarizer")
                elif ("question" in task.lower() or "answer" in task.lower()) and "qa" not in seen_tasks:
                    tasks.append(("qa", {"question": workflow_description, "type": "analytical"}))
                    seen_tasks.add("qa")
                elif "analyze" in task.lower() and "analysis" not in seen_tasks:
                    tasks.append(("analysis", {"topic": workflow_description}))
                    seen_tasks.add("analysis")

            results = []
            for task_type, task_input in tasks:
                result = self.route_task(task_type, task_input)
                results.append(result)

            # Check for conflicts
            if len(results) > 1:
                final_result = self.resolve_conflicts(results)
            else:
                final_result = results[0] if results else {"error": "No tasks executed"}

            workflow_results = {
                "workflow_description": workflow_description,
                "steps_executed": [task_type for task_type, _ in tasks],
                "final_result": final_result
            }
            self.logger.info("🌟 Complex workflow completed!")
            return workflow_results
        except Exception as e:
            self.logger.error(f"❌ Error in workflow: {str(e)}")
            return {"error": str(e)}