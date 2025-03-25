#!/usr/bin/env python3
import json
import os
import random

def generate_synthetic_prompts(count=1000):
    """Generate synthetic prompts that would be asked to Jon"""
    # Define prompt templates based on common conversation topics with Jon
    templates = {
        "relationship": [
            "How are things going with Chelsea?",
            "Have you and Chelsea talked recently?",
            "Any progress with your relationship?",
            "How's married life?",
            "Everything okay at home?",
            "Have you two figured things out?",
            "Are you and Chelsea working things out?",
            "How's the communication going with Chelsea?"
        ],
        "mental_health": [
            "How was your therapy session?",
            "Are you feeling better these days?",
            "Have you been taking care of yourself?",
            "How's your mental health journey going?",
            "Is therapy helping?",
            "Are you still feeling down?",
            "Have you been managing your anxiety?",
            "Are you in a better headspace now?"
        ],
        "work": [
            "How's the new job going?",
            "Any luck with the job search?",
            "Did you hear back from that interview?",
            "Are you still looking for work?",
            "How's work treating you?",
            "Found any good job opportunities?",
            "Are you happy at your current job?",
            "Any progress on the career front?"
        ],
        "creative": [
            "How's the writing coming along?",
            "Made any progress on your book?",
            "Still working on your stories?",
            "Have you been writing lately?",
            "Any new creative projects?",
            "How's the creative work going?",
            "Written anything new?",
            "Still doing the writing thing?"
        ],
        "fitness": [
            "Have you been working out?",
            "Still going to the gym?",
            "How's the fitness journey?",
            "Making progress with your health goals?",
            "Been exercising lately?",
            "Are you still lifting?",
            "How's the workout routine going?",
            "Taking care of your physical health?"
        ],
        "social": [
            "Want to hang out this weekend?",
            "Are you free to get together?",
            "Want to grab food sometime?",
            "Up for gaming tonight?",
            "Want to do something?",
            "Free to catch up soon?",
            "Want to meet up?",
            "Should we plan something?"
        ],
        "check_in": [
            "How have you been?",
            "Everything okay?",
            "Haven't heard from you in a while - how are you?",
            "Just checking in - how are things?",
            "You doing alright?",
            "How's everything going?",
            "What's new with you?",
            "How are you holding up?"
        ],
        "support": [
            "Do you want to talk about it?",
            "Is there anything I can do to help?",
            "Want some company?",
            "Need someone to listen?",
            "Should I come by?",
            "Need to get your mind off things?",
            "Want to get out of the house?",
            "Need anything?"
        ],
        "future": [
            "What are your plans going forward?",
            "Have you thought about what's next?",
            "Where do you see things going?",
            "What do you want to do?",
            "Have you figured out your next steps?",
            "What's your plan?",
            "What are you thinking of doing?",
            "Have you made any decisions?"
        ]
    }

    # Generate prompts with good distribution across categories
    prompts = []
    categories = list(templates.keys())
    
    while len(prompts) < count:
        # Cycle through categories to ensure distribution
        category = categories[len(prompts) % len(categories)]
        
        # Select a random prompt from the category
        category_prompts = templates[category]
        prompt = random.choice(category_prompts)
        
        # Add some natural variation
        if random.random() < 0.2:  # 20% chance to add a follow-up
            followup = random.choice([
                "I've been worried about you.",
                "Just wanted to check in.",
                "No pressure.",
                "If you want to talk.",
                "When you're ready.",
                "I'm here if you need anything.",
                "Take your time.",
                "Let me know."
            ])
            prompt = f"{prompt} {followup}"
        
        if prompt not in prompts:  # Avoid duplicates
            prompts.append(prompt)
    
    return prompts

if __name__ == "__main__":
    prompts = generate_synthetic_prompts(1000)
    print(f"Generated {len(prompts)} prompts")
    
    with open('synthetic_prompts.jsonl', 'w') as f:
        for prompt in prompts:
            f.write(f'"{prompt}"\n')
    
    print(f"Saved prompts to synthetic_prompts.jsonl") 