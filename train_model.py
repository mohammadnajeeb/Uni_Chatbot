"""
Script to fine-tune a small language model for the AMUBot.
"""
import os
import json
import logging
import argparse
from typing import List, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from utils.data_processing import load_text_files, clean_text, extract_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_training_data(raw_dir: str, output_file: str) -> List[Dict[str, str]]:
    """
    Prepare training data for fine-tuning.
    
    Args:
        raw_dir: Directory containing raw text files
        output_file: File to save the processed data
        
    Returns:
        List of formatted conversations
    """
    # Load raw text files
    files = load_text_files(os.path.join(raw_dir, '**/*.txt'))
    logger.info(f"Loaded {len(files)} files for training data preparation")
    
    # Create conversation templates
    conversations = []
    
    for file_path, content in files:
        try:
            # Clean content
            cleaned_content = clean_text(content)
            
            # Extract metadata
            metadata = extract_metadata(content)
            
            # Skip if content is too short
            if len(cleaned_content) < 100:
                continue
            
            # Generate a question based on the content and metadata
            questions = generate_questions(cleaned_content, metadata)
            
            # Create conversation pairs
            for question in questions:
                answer = generate_answer(question, cleaned_content, metadata)
                
                conversation = {
                    "instruction": question,
                    "input": "",
                    "output": answer
                }
                conversations.append(conversation)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Saved {len(conversations)} conversation pairs to {output_file}")
    return conversations

def generate_questions(content: str, metadata: Dict[str, Any]) -> List[str]:
    """
    Generate questions based on content and metadata.
    
    Args:
        content: Cleaned content
        metadata: Metadata dictionary
        
    Returns:
        List of generated questions
    """
    questions = []
    
    # Extract category and title
    category = metadata.get('category', 'general')
    title = metadata.get('title', '')
    
    # Generate basic questions based on category
    if category == 'admission':
        questions.append(f"What are the admission requirements for {title if title else 'AMU'}?")
        questions.append(f"How can I apply for admission to {title if title else 'AMU'}?")
        questions.append(f"What is the last date for {title if title else 'admission applications'}?")
    
    elif category == 'examination':
        questions.append(f"When are the exams for {title if title else 'AMU'} scheduled?")
        questions.append(f"What is the exam pattern for {title if title else 'AMU exams'}?")
        questions.append(f"How can I prepare for {title if title else 'AMU entrance exams'}?")
    
    elif category == 'results':
        questions.append(f"Where can I check my results for {title if title else 'AMU exams'}?")
        questions.append(f"When will the results for {title if title else 'AMU exams'} be announced?")
    
    elif category == 'research':
        questions.append(f"What are the research opportunities in {title if title else 'AMU'}?")
        questions.append(f"How can I apply for a PhD program at {title if title else 'AMU'}?")
    
    elif category == 'courses':
        questions.append(f"What courses are offered by {title if title else 'AMU'}?")
        questions.append(f"Tell me about the {title if title else 'programs'} at AMU")
    
    # Generate general questions
    questions.append(f"What is {title if title else 'AMU'}?")
    questions.append(f"Can you provide information about {title if title else 'Aligarh Muslim University'}?")
    
    return questions[:3]  # Limit to 3 questions per document

def generate_answer(question: str, content: str, metadata: Dict[str, Any]) -> str:
    """
    Generate an answer based on question, content, and metadata.
    
    Args:
        question: Generated question
        content: Cleaned content
        metadata: Metadata dictionary
        
    Returns:
        Generated answer
    """
    # Extract key information
    url = metadata.get('url', '')
    
    # Generate a conversational opening
    openings = [
        "Based on the information from AMU, ",
        "According to AMU's resources, ",
        "Here's what I found about that: ",
        "I'd be happy to help with that. ",
        "That's a good question about AMU. "
    ]
    
    import random
    opening = random.choice(openings)
    
    # Generate the answer
    answer = f"{opening}{content[:500]}"
    
    # Add a closing statement with source reference
    if url:
        answer += f"\n\nFor more detailed information, you can visit: {url}"
    else:
        answer += "\n\nFor the most current information, I recommend checking the official AMU website."
    
    return answer

def fine_tune_model(
    model_name: str,
    training_data_file: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5
):
    """
    Fine-tune a language model.
    
    Args:
        model_name: Base model to fine-tune
        training_data_file: File containing training data
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    logger.info(f"Starting fine-tuning of {model_name}")
    
    # Load training data
    with open(training_data_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    logger.info(f"Loaded {len(training_data)} training examples")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format the training data
    def format_instruction(example):
        return f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
    
    formatted_data = [format_instruction(example) for example in training_data]
    
    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = Dataset.from_dict({"text": formatted_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to=None  # Disable reporting to prevent wandb errors
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save fine-tuned model
    logger.info(f"Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning complete!")

def main():
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model for AMUBot")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw text files")
    parser.add_argument("--output_file", type=str, default="data/training_data.json", help="File to save processed training data")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="data/fine_tuned_model", help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare training data
    if not args.skip_data_prep:
        logger.info("Preparing training data...")
        prepare_training_data(args.raw_dir, args.output_file)
    
    # Fine-tune model
    logger.info("Fine-tuning model...")
    fine_tune_model(
        model_name=args.model_name,
        training_data_file=args.output_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()