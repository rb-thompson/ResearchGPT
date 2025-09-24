from abc import ABC, abstractmethod
import logging

class BaseAgent(ABC):
    def __init__(self, research_assistant):
        """Initialize BaseAgent with research assistant reference."""
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"
        self.logger = logging.getLogger(self.agent_name)
        # Clear existing handlers to prevent duplicates
        self.logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("🟢 BaseAgent initialized! Ready to rock!")

    def _call_mistral(self, prompt: str, temperature: float = None) -> str:
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
    def execute_task(self, task_input):
        raise NotImplementedError("Each agent must implement execute_task method")

class SummarizationAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        self.logger = logging.getLogger(self.agent_name)
        self.logger.handlers = []  # Clear handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("📝 SummarizerAgent ready to summarize!")

    def summarize_document(self, doc_id):
        """Summarize a single document by its ID."""
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

    def create_literature_overview(self, doc_ids):
        """Create a literature overview from multiple document IDs."""
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

    def execute_task(self, task_input):
        """Execute summarization task based on input type."""
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        else:
            self.logger.error("❌ Invalid task input")
            return {"error": "Invalid task input for SummarizerAgent"}
        
class QAAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
        self.logger = logging.getLogger(self.agent_name)
        self.logger.handlers = []  # Clear handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("❓ QAAgent ready to answer questions!")

    def answer_factual_question(self, question):
        """Answer a factual question based on document context."""
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

    def answer_analytical_question(self, question):
        """Answer an analytical question using chain-of-thought reasoning."""
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

    def execute_task(self, task_input):
        """Execute QA task based on question type."""
        question = task_input.get("question", "")
        question_type = task_input.get("type", "factual")
        if question_type == "analytical":
            return self.answer_analytical_question(question)
        else:
            return self.answer_factual_question(question)
        
class AnalysisAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "AnalysisAgent"
        self.logger = logging.getLogger(self.agent_name)
        self.logger.handlers = []  # Clear handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("🔍 AnalysisAgent ready to uncover insights!")

    def execute_task(self, task_input):
        """Execute analysis task based on input topic."""
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
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        self.logger = logging.getLogger(self.agent_name)
        self.logger.handlers = []  # Clear handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.summarizer = SummarizationAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)
        self.analysis_agent = AnalysisAgent(research_assistant)
        self.logger.info("🧠 ResearchWorkflowAgent ready to run research sessions!")

    def conduct_research_session(self, research_topic):
        """Conduct a full research session on a given topic."""
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

    def execute_task(self, task_input):
        """Execute research workflow based on input topic."""
        if "research_topic" in task_input:
            return self.conduct_research_session(task_input["research_topic"])
        else:
            self.logger.error("❌ Invalid task input")
            return {"error": "Invalid task input for ResearchWorkflowAgent"}
        
class AgentOrchestrator:
    def __init__(self, research_assistant):
        self.assistant = research_assistant
        self.logger = logging.getLogger("AgentOrchestrator")
        self.logger.handlers = []  # Clear handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.agents = {
            "summarizer": SummarizationAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "analysis": AnalysisAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant)
        }
        self.shared_memory = {}
        self.logger.info("🎶 AgentOrchestrator ready to conduct the symphony!")

    def route_task(self, task_type, task_input):
        """Route a task to the appropriate agent based on task type."""
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

    def execute_complex_workflow(self, workflow_description):
        """Execute a complex research workflow involving multiple agents."""
        self.logger.info(f"🎬 Executing complex workflow: {workflow_description}")
        try:
            # Simple parsing: assume description specifies tasks
            tasks = []
            if "summarize" in workflow_description.lower():
                tasks.append(("summarizer", {"doc_ids": list(self.assistant.doc_processor.documents.keys())}))
            if "answer" in workflow_description.lower():
                tasks.append(("qa", {"question": workflow_description, "type": "analytical"}))
            if "analyze" in workflow_description.lower():
                tasks.append(("analysis", {"topic": workflow_description}))
            
            results = []
            for task_type, task_input in tasks:
                result = self.route_task(task_type, task_input)
                results.append(result)
            
            # Aggregate results
            aggregate_prompt = f"Combine these results into a coherent response: {results}"
            final_result = self.agents["workflow"]._call_mistral(aggregate_prompt)
            
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