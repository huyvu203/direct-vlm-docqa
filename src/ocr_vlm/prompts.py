"""
Prompt templates for VLM-based DocQA.
Implements the exact prompts from huy_article.md Section "Prompt Design".
"""

from typing import Dict


class PromptTemplate:
    """Manages prompt templates for VLM inference."""
    
    # System message from article
    SYSTEM_MESSAGE = """You are a document analysis assistant. Answer questions about infographics using ONLY information visible in the image. Return JSON format: {"answer": "exact text or number", "confidence": 0.0-1.0}. If unsure, respond with {"answer": "unknown", "confidence": 0.0}.

Few-shot examples:
1. Question: "What is the total revenue?" → {"answer": "$2.4 million", "confidence": 0.95}
2. Question: "Which year had highest growth?" → {"answer": "2023", "confidence": 0.90}"""
    
    # User message template from article
    USER_TEMPLATE = """Question: {question}

Guidelines:
- Extract exact text/numbers from the image
- Do not infer or calculate beyond what's shown
- For percentages, include the % symbol
- For dates, match the format shown

Respond with valid JSON only."""
    
    @classmethod
    def get_system_message(cls) -> str:
        """Get the system message for VLM calls."""
        return cls.SYSTEM_MESSAGE
    
    @classmethod
    def get_user_message(cls, question: str) -> str:
        """
        Get formatted user message for a question.
        
        Args:
            question: Question text
            
        Returns:
            Formatted user prompt
        """
        return cls.USER_TEMPLATE.format(question=question)
    
    @classmethod
    def load_from_file(cls, template_path: str) -> Dict[str, str]:
        """
        Load custom prompt template from file.
        
        Expected file format:
            [SYSTEM]
            System message content...
            
            [USER]
            User template with {question} placeholder...
            
        Returns:
            Dict with 'system' and 'user' keys
        """
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Parse sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.strip() == '[SYSTEM]':
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'system'
                current_content = []
            elif line.strip() == '[USER]':
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'user'
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
