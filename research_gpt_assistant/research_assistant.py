"""
Main ResearchGPT Assistant Class
"""

from mistralai import Mistral
import re

class ResearchGPTAssistant:
    def __init__(self, config, document_processor):
        self.config = config
        self.doc_processor = document_processor
        self.mistral_client = Mistral(api_key=self.config.MISTRAL_API_KEY)
        self.conversation_history = []
        self.prompts = self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        prompts = {
            'chain_of_thought': """
            You are a helpful assistant. Answer the following query using chain-of-thought reasoning.
            Let's think about this step by step...
            Query: {query}
            Context: {context}
            Step 1: Understand the question.
            Step 2: Analyze the context.
            Step 3: Reason towards the answer.
            Final Answer:
            """,
            'self_consistency': """
            Generate 3 diverse reasoning paths to answer the query, then select the most consistent answer.
            Query: {query}
            Context: {context}
            Reasoning Path 1:
            Reasoning Path 2:
            Reasoning Path 3:
            Most consistent answer:
            """,
            'react_research': """
            You are a researcher using ReAct: Thought, Action, Observation.
            Actions available: Search (query documents), Analyze (analyze info), Summarize (summarize findings), Conclude (final answer).
            Query: {query}
            Start with Thought.
            """,
            'document_summary': """
            Summarize the key findings, methodology, and conclusions from the following document chunks:
            Chunks: {chunks}
            Key Findings:
            Methodology:
            Conclusions:
            """,
            'qa_with_context': """
            Answer the query based solely on the provided context from research papers.
            If the answer is not in the context, say 'Information not found'.
            Query: {query}
            Context: {context}
            Answer:
            """,
            'verify_answer': """
            Verify the following answer for relevance, accuracy, and completeness based on the query and context.
            Suggest improvements if needed.
            Query: {query}
            Answer: {answer}
            Context: {context}
            Relevance (1-10):
            Accuracy (1-10):
            Completeness (1-10):
            Suggested Improvements:
            Improved Answer:
            """
        }
        return prompts
    
    def _call_mistral(self, prompt, temperature=None):
        if temperature is None:
            temperature = self.config.TEMPERATURE
        try:
            chat_response = self.mistral_client.chat.complete(
                model=self.config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.config.MAX_TOKENS
            )
            response = chat_response.choices[0].message.content
            self.config.logger.info(f"Mistral API call successful. Tokens used: {chat_response.usage.total_tokens}")
            return response
        except Exception as e:
            self.config.logger.error(f"Error calling Mistral API: {str(e)}")
            return f"Error calling Mistral API: {str(e)}"
    
    def chain_of_thought_reasoning(self, query, context_chunks):
        context = "\n".join([chunk[0] for chunk in context_chunks])
        cot_prompt = self.prompts['chain_of_thought'].format(query=query, context=context)
        response = self._call_mistral(cot_prompt)
        return response
    
    def self_consistency_generate(self, query, context_chunks, num_attempts=3):
        responses = []
        context = "\n".join([chunk[0] for chunk in context_chunks])
        for i in range(num_attempts):
            temp = self.config.TEMPERATURE + (i * 0.1)
            sc_prompt = self.prompts['self_consistency'].format(query=query, context=context)
            response = self._call_mistral(sc_prompt, temperature=temp)
            responses.append(response)
        select_prompt = f"Select the most consistent answer from these {num_attempts} responses:\n" + "\n\n".join(responses) + "\nMost consistent:"
        best_response = self._call_mistral(select_prompt)
        return best_response
    
    def react_research_workflow(self, query):
        workflow_steps = []
        current_prompt = self.prompts['react_research'].format(query=query)
        max_steps = 5
        for step in range(max_steps):
            response = self._call_mistral(current_prompt)
            lines = response.split('\n')
            thought = ""
            action = ""
            for line in lines:
                if line.startswith("Thought:"):
                    thought = line.replace("Thought:", "").strip()
                elif line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
            observation = ""
            if "Search" in action:
                search_query = action.replace("Search", "").strip()
                relevant_chunks = self.doc_processor.find_similar_chunks(search_query, top_k=3)
                observation = "\n".join([chunk[0] for chunk in relevant_chunks])
            elif "Analyze" in action:
                observation = self._call_mistral(f"Analyze: {action.replace('Analyze', '').strip()}")
            elif "Summarize" in action:
                observation = self._call_mistral(f"Summarize: {action.replace('Summarize', '').strip()}")
            elif "Conclude" in action:
                final_answer = action.replace("Conclude", "").strip()
                break
            workflow_steps.append({
                'step': step + 1,
                'thought': thought,
                'action': action,
                'observation': observation
            })
            current_prompt += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}\n"
            if self._should_conclude_workflow(observation):
                break
        if 'final_answer' not in locals():
            final_answer = self._call_mistral(current_prompt + "\nFinal Answer:")
        return {
            'workflow_steps': workflow_steps,
            'final_answer': final_answer
        }
    
    def _should_conclude_workflow(self, observation):
        decide_prompt = f"Does this observation provide sufficient information to conclude? Observation: {observation}\nAnswer yes or no."
        decision = self._call_mistral(decide_prompt)
        return "yes" in decision.lower()
    
    def verify_and_edit_answer(self, answer, original_query, context):
        verify_prompt = self.prompts['verify_answer'].format(query=original_query, answer=answer, context=context)
        verification_result = self._call_mistral(verify_prompt)
        lines = verification_result.split('\n')
        relevance = 0
        accuracy = 0
        completeness = 0
        improvements = ""
        improved_answer = answer
        
        for line in lines:
            if 'Relevance' in line:
                match = re.search(r'(\d+)/10', line)
                relevance = float(match.group(1)) if match else 0
            elif 'Accuracy' in line:
                match = re.search(r'(\d+)/10', line)
                accuracy = float(match.group(1)) if match else 0
            elif 'Completeness' in line:
                match = re.search(r'(\d+)/10', line)
                completeness = float(match.group(1)) if match else 0
            elif line.startswith("Suggested Improvements:"):
                improvements = line.replace("Suggested Improvements:", "").strip()
            elif line.startswith("Improved Answer:"):
                # Extract until next section or end
                idx = lines.index(line)
                improved_answer = "\n".join(lines[idx+1:]).strip()
        verification_data = {
            'original_answer': answer,
            'verification_result': verification_result,
            'improved_answer': improved_answer,
            'confidence_score': (relevance + accuracy + completeness) / 3  # Average 0-10
        }
        return verification_data
    
    def answer_research_question(self, query, use_cot=True, use_verification=True):
        relevant_chunks = self.doc_processor.find_similar_chunks(query, top_k=5)
        if use_cot:
            answer = self.chain_of_thought_reasoning(query, relevant_chunks)
        else:
            context = "\n".join([chunk[0] for chunk in relevant_chunks])
            qa_prompt = self.prompts['qa_with_context'].format(query=query, context=context)
            answer = self._call_mistral(qa_prompt)
        if use_verification:
            context_str = "\n".join([chunk[0] for chunk in relevant_chunks])
            verification_data = self.verify_and_edit_answer(answer, query, context_str)
            final_answer = verification_data['improved_answer']
        else:
            final_answer = answer
            verification_data = None
        response = {
            'query': query,
            'relevant_documents': len(relevant_chunks),
            'answer': final_answer,
            'verification': verification_data,
            'sources_used': [chunk[2] for chunk in relevant_chunks]
        }
        return response