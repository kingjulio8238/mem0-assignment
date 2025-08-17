"""
Memory-Centric Conversational Dataset Generator
Generates 5000 examples for fine-tuning on memory management tasks.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

class MemoryDatasetGenerator:
    def __init__(self):
        self.conversation_templates = {
            "personal_info": [
                "Remember that my name is {name}.",
                "I'm {age} years old.",
                "I live in {location}.",
                "My job is {occupation}.",
                "My birthday is {date}.",
                "I have {siblings} siblings.",
                "My phone number is {phone}.",
                "My email is {email}.",
            ],
            "preferences": [
                "I love {food} food.",
                "My favorite color is {color}.",
                "I prefer {preference} over {alternative}.",
                "I hate {dislike}.",
                "My favorite {category} is {item}.",
                "I'm allergic to {allergen}.",
                "I enjoy {activity} in my free time.",
                "My favorite season is {season}.",
            ],
            "relationships": [
                "My {relation} is named {name}.",
                "{name} is my {relation}.",
                "I'm married to {spouse}.",
                "My best friend is {friend}.",
                "I work with {colleague} at {company}.",
                "My {pet_type} is named {pet_name}.",
                "I'm dating {partner}.",
                "My neighbor {neighbor} is really {trait}.",
            ],
            "facts_and_knowledge": [
                "Remember that {fact}.",
                "I learned that {knowledge}.",
                "Don't forget {important_info}.",
                "The answer to {question} is {answer}.",
                "I need to remember {task} for {date}.",
                "{event} happened on {date}.",
                "The password for {service} is {password}.",
                "My {item} is located in {location}.",
            ],
            "goals_and_plans": [
                "I want to {goal} by {deadline}.",
                "My plan is to {plan}.",
                "I need to {task} tomorrow.",
                "Remember to remind me about {event}.",
                "I'm planning to visit {place} next {timeframe}.",
                "My goal for this year is to {annual_goal}.",
                "I have an appointment with {person} on {date}.",
                "Don't let me forget to {reminder}.",
            ]
        }
        
        self.sample_data = {
            "names": ["Alex", "Jordan", "Casey", "Morgan", "Taylor", "Jamie", "Riley", "Cameron", "Avery", "Quinn"],
            "ages": list(range(18, 80)),
            "locations": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
            "occupations": ["engineer", "teacher", "doctor", "designer", "developer", "writer", "manager", "consultant", "analyst", "researcher"],
            "foods": ["Italian", "Mexican", "Chinese", "Indian", "Thai", "Japanese", "Mediterranean", "French", "Korean", "Vietnamese"],
            "colors": ["blue", "green", "red", "purple", "orange", "yellow", "black", "white", "pink", "gray"],
            "activities": ["reading", "hiking", "cooking", "gaming", "traveling", "photography", "music", "sports", "gardening", "painting"],
            "seasons": ["spring", "summer", "fall", "winter"],
            "relations": ["brother", "sister", "mother", "father", "cousin", "uncle", "aunt", "grandmother", "grandfather"],
            "pet_types": ["dog", "cat", "bird", "fish", "hamster", "rabbit"],
        }

    def generate_conversation(self, memory_type: str) -> Dict[str, str]:
        """Generate a single conversation example."""
        templates = self.conversation_templates[memory_type]
        template = random.choice(templates)
        
        # Fill in template with random data
        filled_template = self._fill_template(template)
        
        # Create conversation format
        user_message = filled_template
        assistant_response = self._generate_assistant_response(memory_type, filled_template)
        
        # Format as conversation
        conversation = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>"
        
        return {
            "text": conversation,
            "memory_type": memory_type,
            "user_input": user_message,
            "assistant_response": assistant_response
        }

    def _fill_template(self, template: str) -> str:
        """Fill template with random sample data."""
        placeholders = {
            "name": random.choice(self.sample_data["names"]),
            "age": random.choice(self.sample_data["ages"]),
            "location": random.choice(self.sample_data["locations"]),
            "occupation": random.choice(self.sample_data["occupations"]),
            "food": random.choice(self.sample_data["foods"]),
            "color": random.choice(self.sample_data["colors"]),
            "activity": random.choice(self.sample_data["activities"]),
            "season": random.choice(self.sample_data["seasons"]),
            "relation": random.choice(self.sample_data["relations"]),
            "pet_type": random.choice(self.sample_data["pet_types"]),
            "date": self._random_date(),
            "phone": self._random_phone(),
            "email": self._random_email(),
        }
        
        # Add more specific placeholders
        placeholders.update({
            "siblings": random.choice(["no", "one", "two", "three"]),
            "preference": random.choice(["coffee", "tea", "morning", "evening", "summer", "winter"]),
            "alternative": random.choice(["tea", "coffee", "evening", "morning", "winter", "summer"]),
            "dislike": random.choice(["spicy food", "cold weather", "loud music", "crowds", "early mornings"]),
            "category": random.choice(["movie", "book", "song", "restaurant", "hobby"]),
            "item": random.choice(["The Matrix", "Pride and Prejudice", "Bohemian Rhapsody", "Pizza Palace", "photography"]),
            "allergen": random.choice(["peanuts", "shellfish", "dairy", "gluten", "cats"]),
            "spouse": random.choice(self.sample_data["names"]),
            "friend": random.choice(self.sample_data["names"]),
            "colleague": random.choice(self.sample_data["names"]),
            "company": random.choice(["TechCorp", "DataSystems", "InnovateInc", "Solutions Ltd", "Global Dynamics"]),
            "pet_name": random.choice(["Buddy", "Luna", "Max", "Bella", "Charlie", "Lucy", "Rocky", "Molly"]),
            "partner": random.choice(self.sample_data["names"]),
            "neighbor": random.choice(self.sample_data["names"]),
            "trait": random.choice(["friendly", "helpful", "quiet", "funny", "kind"]),
            "fact": random.choice([
                "the capital of Australia is Canberra",
                "octopuses have three hearts",
                "the Great Wall of China is not visible from space",
                "honey never spoils",
                "bananas are berries but strawberries aren't"
            ]),
            "knowledge": random.choice([
                "Python is great for data science",
                "exercise improves memory",
                "meditation reduces stress",
                "reading before bed helps sleep",
                "drinking water boosts energy"
            ]),
            "important_info": random.choice([
                "the meeting is at 3 PM",
                "mom's birthday is next week",
                "the car needs an oil change",
                "rent is due on the 1st",
                "the doctor's appointment is on Friday"
            ]),
            "question": random.choice([
                "what's 2+2", "who wrote Romeo and Juliet", "what's the capital of France",
                "when was the first iPhone released", "what does HTTP stand for"
            ]),
            "answer": random.choice(["4", "Shakespeare", "Paris", "2007", "HyperText Transfer Protocol"]),
            "task": random.choice([
                "call the dentist", "buy groceries", "finish the report",
                "water the plants", "pay the bills", "schedule a meeting"
            ]),
            "event": random.choice([
                "my graduation", "the wedding", "our anniversary",
                "the company launch", "my first day at work", "moving day"
            ]),
            "service": random.choice(["email", "bank account", "Netflix", "WiFi", "laptop"]),
            "password": "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)),
            "goal": random.choice([
                "learn Spanish", "run a marathon", "save $10,000",
                "get promoted", "travel to Japan", "write a book"
            ]),
            "deadline": random.choice(["next month", "the end of the year", "my birthday", "summer", "December"]),
            "plan": random.choice([
                "exercise every morning", "read 30 minutes daily",
                "cook at home more often", "learn a new skill", "network more"
            ]),
            "place": random.choice(["Paris", "Tokyo", "London", "New York", "Sydney"]),
            "timeframe": random.choice(["month", "year", "summer", "winter", "spring"]),
            "annual_goal": random.choice([
                "get fit", "learn programming", "start a business",
                "buy a house", "change careers", "travel more"
            ]),
            "person": random.choice(["Dr. Smith", "the lawyer", "my manager", "the accountant"]),
            "reminder": random.choice([
                "take my medicine", "call mom", "backup my files",
                "renew my license", "submit the application", "book vacation"
            ])
        })
        
        try:
            return template.format(**placeholders)
        except KeyError:
            # If we're missing a placeholder, return the template as-is
            return template

    def _generate_assistant_response(self, memory_type: str, user_input: str) -> str:
        """Generate appropriate assistant response based on memory type."""
        responses = {
            "personal_info": [
                "I'll remember that information about you.",
                "Got it, I've stored that personal detail.",
                "Thanks for sharing that with me. I'll keep it in mind.",
                "I've noted that information for future reference.",
                "I'll make sure to remember that about you."
            ],
            "preferences": [
                "I'll remember your preference for future conversations.",
                "Thanks for letting me know what you like/dislike.",
                "I've noted your preference and will keep it in mind.",
                "Good to know! I'll remember that about your tastes.",
                "I'll store that preference information for you."
            ],
            "relationships": [
                "I'll remember that relationship information.",
                "Thanks for telling me about the people in your life.",
                "I've noted that information about your relationships.",
                "Good to know! I'll remember who's important to you.",
                "I'll keep that family/friend information in mind."
            ],
            "facts_and_knowledge": [
                "I'll remember that fact for you.",
                "Thanks for sharing that knowledge with me.",
                "I've stored that information for future reference.",
                "Good to know! I'll keep that fact in mind.",
                "I'll make sure to remember that important detail."
            ],
            "goals_and_plans": [
                "I'll help you remember that goal/plan.",
                "I've noted your plans and will remind you when needed.",
                "Thanks for sharing your goals with me.",
                "I'll keep track of that important task/appointment for you.",
                "I'll make sure to remind you about that when the time comes."
            ]
        }
        
        return random.choice(responses[memory_type])

    def _random_date(self) -> str:
        """Generate a random date string."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now() + timedelta(days=365)
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%B %d, %Y")

    def _random_phone(self) -> str:
        """Generate a random phone number."""
        return f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"

    def _random_email(self) -> str:
        """Generate a random email address."""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "email.com", "example.com"]
        username = random.choice(self.sample_data["names"]).lower()
        number = random.randint(1, 999)
        domain = random.choice(domains)
        return f"{username}{number}@{domain}"

    def generate_dataset(self, total_examples: int = 5000) -> List[Dict[str, str]]:
        """Generate the complete dataset."""
        dataset = []
        memory_types = list(self.conversation_templates.keys())
        examples_per_type = total_examples // len(memory_types)
        
        print(f"Generating {total_examples} examples across {len(memory_types)} memory types...")
        
        for memory_type in memory_types:
            print(f"Generating {examples_per_type} examples for {memory_type}...")
            for _ in range(examples_per_type):
                conversation = self.generate_conversation(memory_type)
                dataset.append(conversation)
        
        # Generate remaining examples to reach exactly 5000
        remaining = total_examples - len(dataset)
        for _ in range(remaining):
            memory_type = random.choice(memory_types)
            conversation = self.generate_conversation(memory_type)
            dataset.append(conversation)
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        print(f"Dataset generation complete! Total examples: {len(dataset)}")
        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]], filepath: str):
        """Save dataset to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Dataset saved to {filepath}")

    def save_training_format(self, dataset: List[Dict[str, str]], filepath: str):
        """Save dataset in format optimized for SFTTrainer (text field only)."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Only save the text field for training
                training_example = {"text": example["text"]}
                json.dump(training_example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Training-format dataset saved to {filepath}")

def main():
    """Generate and save the memory-centric dataset."""
    print("Starting Memory-Centric Dataset Generation...")
    
    generator = MemoryDatasetGenerator()
    dataset = generator.generate_dataset(5000)
    

    # Save training-optimized format (text field only) - this is what we actually need
    generator.save_training_format(dataset, "/home/ubuntu/mem0-assignment/finetune/memory_dataset.jsonl")
    
    # Generate statistics
    memory_type_counts = {}
    for example in dataset:
        memory_type = example["memory_type"]
        memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1
    
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    print("Distribution by memory type:")
    for memory_type, count in memory_type_counts.items():
        print(f"  {memory_type}: {count} examples")
    
    print(f"\nDataset saved as 'memory_dataset.jsonl' with text field only (ready for SFTTrainer)")
    print("File format: Each line contains {\"text\": \"<conversation>\"}")
    
    # Show sample conversations
    print("\nSample conversations:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1} ({example['memory_type']}):")
        print(example['text'])
        print("-" * 50)

if __name__ == "__main__":
    main()
