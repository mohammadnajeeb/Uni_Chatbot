"""
Core Faiz Chatbot implementation for answering user queries using Hugging Face models.
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import utilities
from utils.vector_store import VectorStore

# Import LangChain components
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import Hugging Face components directly
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class FaizChatbot:
    """Main chatbot implementation using Hugging Face models."""
    
    def __init__(
        self,
        vector_db_path: str = None,
        model_name: str = None,
        embedding_model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the Faiz chatbot.
        
        Args:
            vector_db_path: Path to the vector database
            model_name: Hugging Face model to use for chat
            embedding_model: Hugging Face model to use for embeddings
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in the response
        """
        # Set default vector DB path if not provided
        if vector_db_path is None:
            vector_db_path = os.path.join('data', 'vectordb')
            
        # Set default model names from environment variables if not provided
        if model_name is None:
            model_name = os.getenv('CHAT_MODEL', 'google/flan-t5-base')
            
        if embedding_model is None:
            embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Store parameters
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
            
        # Initialize Hugging Face embeddings
        logger.info(f"Initializing embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize the vector store
        try:
            self.vector_store = VectorStore(vector_db_path)
            logger.info(f"Vector store initialized with {self.vector_store.count()} documents")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
            
        # Define prompt templates
        self.qa_template = """
        You are Faiz, a helpful, professional, yet conversational chatbot for AMU (Aligarh Muslim University). 
        You are designed to provide accurate and helpful information about AMU and its examination system.
        
        Your personality:
        - Professional and knowledgeable but also warm and engaging
        - Occasionally use light humor when appropriate
        - Always be respectful and considerate
        - Use concise language but be thorough in your explanations
        - Occasionally use Urdu/Hindi phrases like "Talaba" (students), "Jaamia" (university), etc.
        
        When answering:
        - If you don't know something, admit it and don't make things up
        - Always base your answers on the provided context and your knowledge about AMU
        - Format your responses in a readable way using markdown when helpful
        - For questions about exams, courses, admission, or campus, prioritize the provided context
        - When citing specific information, mention the source when available
        
        CONTEXT INFORMATION:
        {context}
        
        QUESTION: {question}
        
        YOUR RESPONSE:
        """
        
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents in the vector store.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.search(
            query=query,
            embed_fn=self.embeddings.embed_documents,
            n_results=n_results
        )
        
    def generate_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format the document
            source = f"[{doc['metadata'].get('url', 'Unknown source')}]"
            score = f"(Relevance: {doc['score']:.2f})"
            
            context_part = f"DOCUMENT {i+1} {source} {score}:\n{doc['text']}\n"
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
        
    def answer(self, question: str) -> str:
        """
        Generate an answer to the user's question.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer
        """
        # Handle greetings more naturally
        greeting_keywords = ["hello", "hi", "hey", "greetings", "assalamu alaikum", "salam"]
        if question.lower().strip() in greeting_keywords or question.lower().strip().startswith(tuple(greeting_keywords)):
            return """Hello! I'm AMUBot, your Aligarh Muslim University assistant. 

I can help you with information about:
- Admission procedures
- Exam schedules and results
- Course offerings
- PhD programs
- Application processes
- And more related to AMU

How can I assist you today?"""
        
        try:
            # Search for relevant documents
            documents = self.search_documents(question)
            
            if not documents:
                return "I couldn't find any information about that in my knowledge base. Please try asking a different question about AMU or its examination system."
            
            # Generate useful context from the documents
            clean_context = ""
            for i, doc in enumerate(documents[:3]):
                # Clean the text to handle encoding issues
                text = doc['text']
                text = ''.join(char for char in text if ord(char) >= 32 and ord(char) < 127)
                
                if len(text.strip()) > 50:
                    clean_context += f"DOCUMENT {i+1}: {text}\n\n"
            
            if not clean_context:
                return """I found some information related to your question, but it appears to have formatting issues.

Please try to rephrase your question or ask about a specific topic like:
- Admission procedures for 2025-26
- PhD application process
- Exam schedules
- Answer keys for recent tests"""
            
            # Create prompt for language model
            prompt = f"""Based on the following information about Aligarh Muslim University (AMU), please answer the user's question in a helpful, informative, and conversational way.

USER QUESTION: {question}

INFORMATION:
{clean_context}

Your response should be:
1. Conversational and friendly in tone
2. Directly addressing the user's question
3. Concise but thorough
4. Well-structured with paragraphs and bullet points when appropriate
5. Include relevant dates and deadlines if present in the information
6. Mention that you're providing information based on AMU's official resources
7. Encourage the user to visit the official AMU website for the most up-to-date information

RESPONSE:"""
            
            try:
                # Try using a simpler LLM approach
                response = self._generate_response_with_template(prompt, clean_context, question)
                return response
            except Exception as e:
                logger.error(f"Error with language model: {e}")
                # Fallback to returning formatted document excerpts
                response = "Here's what I found about your question:\n\n"
                
                for i, doc in enumerate(documents[:3]):
                    source = doc['metadata'].get('url', 'AMU Document')
                    
                    # Clean the text to handle encoding issues
                    text = doc['text']
                    text = ''.join(char for char in text if ord(char) >= 32 and ord(char) < 127)
                    
                    if len(text.strip()) < 50:
                        continue
                    
                    response += f"**Source {i+1}**: {source}\n\n"
                    response += f"{text[:500]}...\n\n"
                
                response += "\nThese are excerpts from the AMU Controller of Exams website. For more detailed information, please visit the official website at amucontrollerexams.com."
                
                return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "I'm sorry, but I encountered an error while processing your question. Please try again with a different question about AMU."
    
    def _generate_response_with_template(self, prompt, context, question):
        """Generate a response using templates for common question patterns."""
        # Check for common question patterns and use templates
        question_lower = question.lower()
        
        # Application/Admission dates template
        if any(term in question_lower for term in ["admission date", "application date", "last date", "deadline", "when can i apply"]):
            return self._format_dates_response(context)
            
        # Process for application template
        elif any(term in question_lower for term in ["how to apply", "application process", "how can i apply", "steps to apply"]):
            return self._format_application_process(context)
            
        # Results/Answer Keys template
        elif any(term in question_lower for term in ["result", "answer key", "score", "marks"]):
            return self._format_results_response(context)
            
        # Document requirements template
        elif any(term in question_lower for term in ["document", "certificate", "require", "eligibility"]):
            return self._format_document_requirements(context)
            
        # Fallback to using a local model or rule-based response
        else:
            # Extract key information from context
            important_sentences = self._extract_important_sentences(context, question)
            return self._craft_response(important_sentences, question)
    
    def _extract_important_sentences(self, context, question):
        """Extract sentences from context that are most relevant to the question."""
        import re
        from collections import Counter
        
        # Tokenize context into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Extract key terms from question
        question_terms = re.findall(r'\b\w+\b', question.lower())
        question_terms = [term for term in question_terms if len(term) > 3]  # Filter out short words
        
        # Score sentences based on relevance to question
        scored_sentences = []
        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            sentence_terms = re.findall(r'\b\w+\b', sentence.lower())
            matches = sum(1 for term in question_terms if term in sentence_terms)
            score = matches / (len(sentence_terms) + 0.1)  # Normalize by sentence length
            
            # Boost sentences with dates, deadlines, or other important information
            important_patterns = ["date", "deadline", "apply", "submit", "form", "document", "require", "procedure", "process"]
            for pattern in important_patterns:
                if pattern in sentence.lower():
                    score += 0.2
                    
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sentences[:5]]  # Return top 5 most relevant sentences
    
    def _craft_response(self, important_sentences, question):
        """Craft a human-like response based on important sentences and the question."""
        if not important_sentences:
            return "I'm sorry, I couldn't find specific information about that in the AMU resources. Please check the official AMU website or try rephrasing your question."
        
        # Detect what type of information the user is looking for
        question_lower = question.lower()
        response_parts = []
        
        # Opening
        if "admission" in question_lower or "apply" in question_lower:
            response_parts.append("Regarding your question about admissions at AMU, here's what I found:")
        elif "exam" in question_lower or "test" in question_lower:
            response_parts.append("About the exams you're asking about at AMU, I can share this information:")
        elif "course" in question_lower or "program" in question_lower:
            response_parts.append("Based on AMU's information about their courses and programs:")
        elif "phd" in question_lower or "research" in question_lower:
            response_parts.append("Regarding PhD programs at AMU, here's what I found:")
        else:
            response_parts.append("Here's what I found about your question regarding AMU:")
        
        # Main content
        response_parts.append("\n\n")
        for sentence in important_sentences:
            clean_sentence = sentence.strip()
            if clean_sentence:
                response_parts.append(f"{clean_sentence}")
        
        # Closing
        response_parts.append("\n\nThis information is based on AMU's official resources. For the most current and complete details, I recommend visiting the official AMU website or contacting the university directly.")
        
        return "".join(response_parts)
        
    def _format_dates_response(self, context):
        """Format a response about application/admission dates."""
        # Extract dates using regex
        import re
        date_patterns = [
            r'\b\d{1,2}(?:st|nd|rd|th)? [A-Z][a-z]+ \d{4}\b',  # e.g., "21st April 2025"
            r'\b[A-Z][a-z]+ \d{1,2}(?:st|nd|rd)?, \d{4}\b',  # e.g., "April 21st, 2025"
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # e.g., "21/04/2025"
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # e.g., "21-04-2025"
            r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b',  # e.g., "21.04.2025"
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, context))
        
        if not dates:
            # Fallback to extracting important sentences
            important_sentences = self._extract_important_sentences(context, "application dates deadline")
            return self._craft_response(important_sentences, "admission dates")
        
        response = "Based on the information from AMU, here are the important dates related to admissions:\n\n"
        
        # Look for sentences containing these dates for more context
        date_contexts = []
        sentences = re.split(r'(?<=[.!?])\s+', context)
        for sentence in sentences:
            for date in dates:
                if date in sentence:
                    date_contexts.append(sentence.strip())
                    break
        
        if date_contexts:
            for context in date_contexts[:5]:  # Limit to 5 to avoid overwhelming
                response += f"• {context}\n"
        else:
            response += "• " + "\n• ".join(dates[:5])
        
        response += "\n\nPlease note that these dates are subject to change. For the most up-to-date information, visit the official AMU website or contact the university directly."
        
        return response
    
    def _format_application_process(self, context):
        """Format a response about application process."""
        # Extract sentences that appear to be describing steps
        import re
        step_patterns = [
            r'(?i)step \d+[:.]\s*[A-Z][^.!?]*[.!?]',  # e.g., "Step 1: Go to the website."
            r'(?i)(?:first|second|third|fourth|fifth)[,:\s]+[A-Z][^.!?]*[.!?]',  # e.g., "First, go to the website."
            r'(?i)(?:i|ii|iii|iv|v|vi)[).:]\s*[A-Z][^.!?]*[.!?]',  # e.g., "i) Go to the website."
            r'(?i)\d+[).]\s*[A-Z][^.!?]*[.!?]',  # e.g., "1. Go to the website."
        ]
        
        steps = []
        for pattern in step_patterns:
            steps.extend(re.findall(pattern, context))
        
        if not steps:
            # Fall back to extracting important sentences
            important_sentences = self._extract_important_sentences(context, "how to apply application process")
            return self._craft_response(important_sentences, "application process")
        
        response = "Here's the application process for AMU based on the information available:\n\n"
        
        for i, step in enumerate(steps[:10]):  # Limit to 10 steps
            response += f"{i+1}. {step.strip()}\n"
        
        response += "\n\nTo ensure you have the most accurate and current application process, please visit the official AMU website or contact the admissions office directly."
        
        return response
    
    def _format_results_response(self, context):
        """Format a response about results or answer keys."""
        important_sentences = self._extract_important_sentences(context, "results answer keys scores")
        
        if not important_sentences:
            return "I couldn't find specific information about results or answer keys in the available data. Please check the official AMU website for the most up-to-date information on results and answer keys."
        
        response = "Regarding the results or answer keys you're asking about, here's what I found from AMU:\n\n"
        
        for sentence in important_sentences:
            response += f"• {sentence.strip()}\n"
        
        response += "\n\nFor accessing results and answer keys, you typically need to visit the official AMU Controller of Examinations website (amucontrollerexams.com) and look for the relevant links. You may need your enrollment number or other credentials to access your specific results."
        
        return response
    
    def _format_document_requirements(self, context):
        """Format a response about document requirements."""
        important_sentences = self._extract_important_sentences(context, "documents required eligibility criteria")
        
        if not important_sentences:
            return "I couldn't find specific information about document requirements in the available data. For the most accurate list of required documents, please check the official AMU website or contact the admissions office."
        
        response = "Based on AMU's information, here are the document requirements you should be aware of:\n\n"
        
        for sentence in important_sentences:
            response += f"• {sentence.strip()}\n"
        
        response += "\n\nMake sure to prepare all required documents before the application deadline. For the complete and most current list of requirements, please refer to the official AMU website or contact the university directly."
        
        return response