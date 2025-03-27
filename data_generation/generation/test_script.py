#!/usr/bin/env python3
"""Test script to debug issues with the jon_data_generator.py file."""

# Mock data structure to simulate ATTACHMENT_PATTERNS
ATTACHMENT_PATTERNS = {
    "defense_mechanisms": [
        {"name": "humor", "indicators": ["haha", "lol", "just kidding"]},
        {"name": "denial", "indicators": ["it's fine", "whatever", "doesn't matter"]},
    ],
    "anxious_behaviors": [
        "excessive reassurance seeking",
        "fear of abandonment",
        "hyper-vigilance",
        "reading into things"
    ]
}

def test_defense_mechanisms():
    """Test handling of defense mechanisms."""
    text = "I'm just kidding, it's fine, whatever haha"
    text_lower = text.lower()
    
    identified = []
    
    # Test with dictionary structure
    for defense in ATTACHMENT_PATTERNS["defense_mechanisms"]:
        if isinstance(defense, dict):
            defense_name = defense.get("name", "")
            indicators = defense.get("indicators", [])
            
            # Check if any indicators appear in the text
            matches = [ind for ind in indicators if ind.lower() in text_lower]
            if matches:
                identified.append({"name": defense_name, "matches": matches})
    
    print("Defense Mechanisms Test:")
    print(f"  Text: '{text}'")
    print(f"  Identified: {identified}")

def test_anxious_behaviors():
    """Test handling of anxious behaviors."""
    text = "I'm just worried you might leave. Are we okay? I need some reassurance."
    text_lower = text.lower()
    
    identified = []
    
    # Test with string list structure - original method
    print("\nAnxious Behaviors Test - Original Method:")
    for indicator in ATTACHMENT_PATTERNS["anxious_behaviors"]:
        # Simple string check
        if isinstance(indicator, str) and indicator.lower() in text_lower:
            identified.append(indicator)
    
    print(f"  Text: '{text}'")
    print(f"  Identified: {identified}")
    
    # Test with improved method - check for partial matches using word tokens
    identified_improved = []
    for indicator in ATTACHMENT_PATTERNS["anxious_behaviors"]:
        if not isinstance(indicator, str):
            continue
            
        # Split into words for fuzzy matching
        indicator_words = indicator.lower().split()
        if any(word in text_lower for word in indicator_words if len(word) > 3):
            identified_improved.append(indicator)
    
    print("\nAnxious Behaviors Test - Improved Method:")
    print(f"  Text: '{text}'")
    print(f"  Identified: {identified_improved}")
    
    # Test with separate keywords approach
    keywords = {
        "excessive reassurance seeking": ["reassurance", "worried", "okay"],
        "fear of abandonment": ["leave", "abandon", "alone"],
        "hyper-vigilance": ["vigilant", "watch", "monitoring"],
        "reading into things": ["reading into", "assume", "assuming"]
    }
    
    identified_keywords = []
    for behavior, terms in keywords.items():
        if any(term in text_lower for term in terms):
            identified_keywords.append(behavior)
    
    print("\nAnxious Behaviors Test - Keywords Method:")
    print(f"  Text: '{text}'")
    print(f"  Identified: {identified_keywords}")

if __name__ == "__main__":
    print("Running Jon Data Generator debugging tests...")
    test_defense_mechanisms()
    test_anxious_behaviors()
    print("\nTests completed.") 