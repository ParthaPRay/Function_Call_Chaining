# This curl caller code calls the simple functional call xecuted by 'chain_functaion_call_53.py'
# Partha Pratim Ray
# 15 November, 2024


import requests
import json

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# Two test prompts for each function
# List of prompts with exactly two function instructions
prompts_with_two_functions = [
    "Plot air quality of Delhi",
    "Can you plot air quality data of Kolkata?",
    "Kindly plot air quality data of Los Angeles",
    "Plot air quality of Mumbai",
    "Could you plot air quality data of Perth?",
    "Kindly plot air quality data of Pune",
    "Plot air quality of Gangtok",
    "Can you plot air quality data of Hyderabad?",
    "Kindly plot air quality data of Bhubaneswar",
    "Do plot air quality data of Bengaluru"
]

# Double the number of functions in the server code
# List of irrelevant prompts to get 'no_route' response 
irrelevant_prompts = [
    "Find addition of 5 and 6",
    "What to calculate boiling point of water?",
    "How do I fix a leaky faucet?",
    "Who painted the Mona Lisa?",
    "Tell me about black holes",
    "How do I get to the nearest gas station?",
    "Can you help me with my homework?",
    "How many continents are there?",
    "What are the symptoms of the common cold?",
    "Tell me a story"
]

    # "Tell me about black holes."
    # "How do I get to the nearest gas station?",
    # "Can you help me with my homework?",
    # "How many continents are there?",
    # "What are the symptoms of the common cold?",
    # "How do I change my email password?",
    # "Tell me a story.",   
    # "How do I set up a new printer?",
    # "What is the best way to learn a new language?",
    # "What is the difference between a noun and a verb?",
    # "Explain Newton laws of motion.",
    # "What is the capital of Egypt?",
    # "What is the meaning of the word serendipity?",
    # "What is the recipe for pancakes?",
    # "Who is the author of 1984 book?",
    # "What is the latest score in the baseball game?" 
    # "What is the speed of light?",
    # "How do you say hello in Spanish?",    
    # "What is on TV tonight?",
    # "Tell me the plot of Hamlet.",
    # "Can you recommend a good book?",
    # "Explain the Pythagorean theorem.",
    # "Who won the Nobel Prize in Physics in 2020?"    
    # "Explain quantum physics"
    # "Who won the football match yesterday?"
    # "How do airplanes fly?"
    # "Define the meaning of life"

# Combine the prompts
all_prompts = prompts_with_two_functions + irrelevant_prompts

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        result = response.json()
        return (prompt, result)
    except Exception as e:
        return (prompt, f"Error: {str(e)}")

# Main function to execute the script
if __name__ == "__main__":
    # Loop through each prompt and send to the API
    for prompt in all_prompts:
        prompt_text, response = send_prompt(prompt)
        print(f"Prompt: {prompt_text}")
        print(f"Response: {json.dumps(response, indent=2)}\n")
