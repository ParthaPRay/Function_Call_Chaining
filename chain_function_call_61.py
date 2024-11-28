# Simple Function Calling
# Partha Pratim Ray, Sikkim University
# 15/11/2024
#
# Configuration 
# shot
# Functions: 4
# First calls get_coordinates() to get lat and lon of the city and then calls the get_air_data() with these lat and lon to get current air quality data and then plot_air_data() and then size_plot_file()


# api for openweathermap:  4a265265d7ea421d0cc3f782ad8ba67e

# Curl commands for testing:

# External API Interaction

# Get coordinates: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Get the coordinates of Los Angeles."}'


###################### Works good fast

############## Three functions, get_coordinates, get_air_data, and plot_air_data

#### Saves plots in varied file name


from fastapi import FastAPI
import numpy as np
import requests
import threading
import psutil
import time
import csv
import os
from pydantic import BaseModel
from queue import Queue
from statistics import mean
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Added imports for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()

# Define the threshold for similarity score
THRESHOLD = 0.4950  # Adjust based on embedding model

# Define the embedding model and LLM
embed_model = "all-minilm:33m"  # Embedding model
model_name = "smollm2:1.7b-instruct-q4_K_M"  # LLM for dynamic routes # qwen2.5:0.5b-instruct  # llama3.2:1b-instruct-q4_K_M # smollm2:1.7b-instruct-q4_K_M
OLLAMA_API_URL = "http://localhost:11434/api/embed"
OLLAMA_LLM_URL = "http://localhost:11434/api/chat"

# CSV file setup
csv_file = 'chain_function_call_logs.csv'
csv_headers = [
    'timestamp', 'model_name', 'embed_model', 'prompt', 'response', 'route_type', 'route_selected',
    'semantic_similarity_score', 'similarity_metric', 'vector', 'total_duration',
    'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count',
    'eval_duration', 'tokens_per_second', 'avg_cpu_usage_during', 'memory_usage_before',
    'memory_usage_after', 'memory_allocated_for_model', 'network_latency', 'total_response_time',
    'route_selection_duration', 'llm_invoked', 'function_execution_time_total_ns',
    'function_execution_times_ns', 'llm_response_parsing_duration_ns', 'number_of_functions_called'
]

csv_queue = Queue()
cpu_usage_queue = Queue()
memory_usage_queue = Queue()
is_monitoring = False
memory_allocated_for_model = 0

# CSV writer thread to log the data
def csv_writer():
    while True:
        log_message_csv = csv_queue.get()
        if log_message_csv is None:  # Exit signal
            break
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_headers)
            writer.writerow(log_message_csv)

csv_thread = threading.Thread(target=csv_writer)
csv_thread.start()

class Prompt(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    # Measure memory usage for the model
    global memory_allocated_for_model
    memory_allocated_for_model = load_model_and_measure_memory(model_name)
    print(f"Memory Allocated for Model '{model_name}': {memory_allocated_for_model / (1024 * 1024):.2f} MB")

# Function to measure memory allocated after loading the model
def load_model_and_measure_memory(model_name):
    # Load the model by making an empty prompt request
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False
    }
    response = requests.post(OLLAMA_LLM_URL, json=payload)
    if response.status_code == 200:
        print(f"Model '{model_name}' loaded successfully.")
    else:
        print(f"Failed to load model '{model_name}'. Response: {response.text}")

    # Get the list of loaded models using the /api/ps endpoint
    ps_response = requests.get("http://localhost:11434/api/ps")
    if ps_response.status_code == 200:
        models_info = ps_response.json().get('models', [])
        for model_info in models_info:
            if model_info.get('name') == model_name:
                model_size = model_info.get('size', 0)
                return model_size  # Size is in bytes
        print(f"Model '{model_name}' not found in the loaded models.")
        return 0
    else:
        print(f"Failed to retrieve models using /api/ps. Response: {ps_response.text}")
        return 0

# Modified Route class to include functions and function_schemas
class Route:
    def __init__(self, name, utterances, dynamic=False, function_schemas=None, functions=None):
        self.name = name
        self.utterances = utterances
        self.dynamic = dynamic
        self.function_schemas = function_schemas or []  # For dynamic routes
        self.functions = functions or []  # For dynamic routes

# Define functions for dynamic routes

def get_coordinates(city_name: str) -> dict:
    """Retrieve the latitude and longitude for a specified city using the OpenWeatherMap Geocoding API."""
    try:
        # Replace with your actual API key
        api_key = "4a265265d7ea421d0cc3f782ad8ba67e"
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = data[0]["lat"]
                lon = data[0]["lon"]
                return {"latitude": lat, "longitude": lon}
            else:
                return {"error": f"No data found for the city: {city_name}"}
        else:
            return {"error": f"Error: Unable to fetch data for {city_name}. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_air_data(lat: float, lon: float) -> dict:
    """Fetch the current air pollution data for specified coordinates."""
    try:
        # Replace with your actual API key
        api_key = "4a265265d7ea421d0cc3f782ad8ba67e"
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data  # Return the JSON response
        else:
            return {"error": f"Error fetching air data. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def plot_air_data(air_data: dict, city_name: str) -> str:
    """Plot air quality components using matplotlib, seaborn, and pandas."""
    try:
        # Extract components from air_data
        if 'list' in air_data and air_data['list']:
            components = air_data['list'][0]['components']
            # Create DataFrame
            df = pd.DataFrame(list(components.items()), columns=['Component', 'Concentration'])
            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            # Plot using seaborn
            plt.figure(figsize=(10, 6))
            # Use a color palette with as many colors as components
            palette = sns.color_palette("husl", len(df))
            ax = sns.barplot(x='Component', y='Concentration', data=df, palette=palette)
            # Set the title
            plt.title(f'Air Quality Components of {city_name} on {current_date}')
            plt.xlabel('Component')
            plt.ylabel('Concentration (μg/m³)')
            # Add values on top of each bar
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
            plt.tight_layout()
            # Save plot to file with varying filename
            filename_safe_city_name = city_name.replace(" ", "_")
            output_file = f"{filename_safe_city_name}_{current_date}.png"
            plt.savefig(output_file)
            plt.close()
            # Return the filename
            return output_file
        else:
            return "No air quality data available to plot."
    except Exception as e:
        return f"Error plotting air data: {str(e)}"

def size_plot_file(filename: str) -> dict:
    """Measure the size of the plot file in kilobytes."""
    try:
        file_size = os.path.getsize(filename) / 1024  # size in kilobytes
        return {"file_size_kb": file_size}
    except Exception as e:
        return {"error": str(e)}

# Define function schemas

get_coordinates_schema = {
    "name": "get_coordinates",
    "description": "Get the latitude and longitude of a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "The name of the city."
            }
        },
        "required": ["city_name"]
    }
}

get_air_data_schema = {
    "name": "get_air_data",
    "description": "Fetch current air pollution data for specified coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "Latitude of the location."
            },
            "lon": {
                "type": "number",
                "description": "Longitude of the location."
            }
        },
        "required": ["lat", "lon"]
    }
}

plot_air_data_schema = {
    "name": "plot_air_data",
    "description": "Plot air quality components from air data.",
    "parameters": {
        "type": "object",
        "properties": {
            "air_data": {
                "type": "object",
                "description": "The air data dictionary obtained from get_air_data."
            },
            "city_name": {
                "type": "string",
                "description": "The name of the city."
            }
        },
        "required": ["air_data", "city_name"]
    }
}

size_plot_file_schema = {
    "name": "size_plot_file",
    "description": "Measure the size of the plot file in kilobytes.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The name of the plot file."
            }
        },
        "required": ["filename"]
    }
}

# Define routes with dynamic options
routes = [
    Route(
        name="dynamic_functions",
        utterances=[
            "Plot air quality of London and measure the plot file size",
            "Can you plot air quality data of Delhi and tell me the file size?",
            "Kindly plot air quality data of Kolkata and measure the size of the plot file",
        ],
        dynamic=True,
        functions=[get_coordinates, get_air_data, plot_air_data, size_plot_file],
        function_schemas=[get_coordinates_schema, get_air_data_schema, plot_air_data_schema, size_plot_file_schema]
    )
]

# Function to create multi-shot prompt for the LLM
def create_multi_shot_prompt(prompt: str, route: Route) -> str:
    examples = """
Example 1:
User: Plot air quality of Sydney and measure the plot file size
Assistant:
Function: get_coordinates
Arguments: {"city_name": "Sydney"}
Result: {"latitude": -33.8688, "longitude": 151.2093}
Function: get_air_data
Arguments: {"lat": -33.8688, "lon": 151.2093}
Result: {"coord": {"lon": 151.2093, "lat": -33.8688}, "list": [...]}
Function: plot_air_data
Arguments: {"air_data": "$get_air_data", "city_name": "Sydney"}
Result: "Sydney_2024-11-15.png"
Function: size_plot_file
Arguments: {"filename": "$plot_air_data"}
Result: {"file_size_kb": 102.4}

Example 2:
User: Plot air quality of New York and tell me the file size
Assistant:
Function: get_coordinates
Arguments: {"city_name": "New York"}
Result: {"latitude": 40.7128, "longitude": -74.0060}
Function: get_air_data
Arguments: {"lat": 40.7128, "lon": -74.0060}
Result: {"coord": {"lon": -74.0060, "lat": 40.7128}, "list": [...]}
Function: plot_air_data
Arguments: {"air_data": "$get_air_data", "city_name": "New York"}
Result: "New_York_2024-11-15.png"
Function: size_plot_file
Arguments: {"filename": "$plot_air_data"}
Result: {"file_size_kb": 98.7}

Example 3:
User: Can you plot air quality of Delhi and measure the file size?
Assistant:
Function: get_coordinates
Arguments: {"city_name": "Delhi"}
Result: {"latitude": 28.6517, "longitude": 77.2219}
Function: get_air_data
Arguments: {"lat": 28.6517, "lon": 77.2219}
Result: {"coord": {"lon": 77.2219, "lat": 28.6517}, "list": [...]}
Function: plot_air_data
Arguments: {"air_data": "$get_air_data", "city_name": "Delhi"}
Result: "Delhi_2024-11-15.png"
Function: size_plot_file
Arguments: {"filename": "$plot_air_data"}
Result: {"file_size_kb": 105.3}

"""
    prompt_template = examples + f"""
Now, respond to this query:
User: {prompt}
Assistant:
"""
    return prompt_template

# The rest of the code remains the same (monitoring functions, LLM invocation, etc.)
# ...

# Resource monitoring thread to track CPU and memory usage
def monitor_resources():
    global is_monitoring
    process = psutil.Process()
    while is_monitoring:
        cpu_usage = psutil.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss  # Memory in bytes
        cpu_usage_queue.put(cpu_usage)
        memory_usage_queue.put(memory_usage)
        time.sleep(0.01)  # Poll every 10ms

# Function to call LLM with multi-shot prompt and extract metrics
def call_llm_with_multi_shot(prompt, route):
    response = requests.post(
        OLLAMA_LLM_URL,
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
    )
    
    response_json = response.json()

    # Check if the expected keys are present in the response
    if 'message' in response_json and 'content' in response_json['message']:
        generated_response = response_json['message']['content']
    else:
        generated_response = "Error: Unexpected response structure from LLM"

    # Add debug print to log the LLM's raw output
    print(f"LLM Generated Response:\n{generated_response}\n")

    # Measure the time to parse LLM output
    parse_start_time = time.time()
    # Parse the LLM's response to extract function calls and arguments
    function_calls = parse_llm_response(generated_response, route.functions)
    parse_end_time = time.time()
    llm_response_parsing_duration = (parse_end_time - parse_start_time) * 1e9  # Convert to nanoseconds

    response_texts = []
    
    # Initialize list to hold function execution times
    function_execution_times = []
    
    # Number of functions called
    number_of_functions_called = len(function_calls)
    
    # Context to store results of functions
    context = {}

    # Process each function call in sequence
    for function, arguments in function_calls:
        if function and arguments is not None:
            # Before executing the function, check if arguments reference previous results
            # Replace argument values if they refer to previous results
            for arg_name, arg_value in arguments.items():
                if isinstance(arg_value, str) and arg_value.startswith("$"):
                    # It's a reference to a previous result
                    var_name = arg_value[1:]  # Remove the leading $
                    if var_name in context:
                        arguments[arg_name] = context[var_name]
                    else:
                        # Variable not found, cannot proceed
                        response_texts.append(f"Error: Variable '{var_name}' not found in context.")
                        continue  # Skip to next function call

            # Measure the function execution time
            func_start_time = time.time()
            try:
                result = function(**arguments)
                func_end_time = time.time()
                execution_time = (func_end_time - func_start_time) * 1e9  # Convert to nanoseconds
                function_execution_times.append({
                    'function_name': function.__name__,
                    'execution_time_ns': execution_time
                })
                # Store the result in the context with the function name as the key
                context[function.__name__] = result

                # Format the result for better readability
                formatted_result = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

                response_texts.append(f"Function: {function.__name__}\nResult: {formatted_result}")
            except Exception as e:
                func_end_time = time.time()
                execution_time = (func_end_time - func_start_time) * 1e9  # Convert to nanoseconds
                function_execution_times.append({
                    'function_name': function.__name__,
                    'execution_time_ns': execution_time
                })
                response_texts.append(f"Function: {function.__name__}\nError executing function: {e}")
        else:
            response_texts.append(f"Could not parse function call from LLM response.")

    # Combine results from all tasks
    response_text = '\n'.join(response_texts)

    # Extract the relevant metrics
    total_duration = response_json.get('total_duration', 0)
    load_duration = response_json.get('load_duration', 0)
    prompt_eval_count = response_json.get('prompt_eval_count', 0)
    prompt_eval_duration = response_json.get('prompt_eval_duration', 0)
    eval_count = response_json.get('eval_count', 0)
    eval_duration = response_json.get('eval_duration', 1)  # Avoid division by zero

    # Return the response and extracted metrics
    return {
        "generated_response": response_text,
        "metrics": {
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "function_execution_times": function_execution_times,
            "llm_response_parsing_duration_ns": llm_response_parsing_duration,
            "number_of_functions_called": number_of_functions_called
        }
    }

# Function to parse LLM response and map to valid functions
def parse_llm_response(response_text, available_functions):
    # Extract multiple function calls from the LLM response
    lines = response_text.strip().split('\n')
    function_calls = []
    function = None
    arguments = None

    # Map function names to the actual function objects
    function_mapping = {func.__name__: func for func in available_functions}

    for line in lines:
        if line.startswith('Function:'):
            if function and arguments is not None:
                function_calls.append((function, arguments))
            raw_function_name = line[len('Function:'):].strip()
            function = function_mapping.get(raw_function_name, None)
            arguments = None
        elif line.startswith('Arguments:'):
            args_text = line[len('Arguments:'):].strip()
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                arguments = None
        elif line.startswith('Result:'):
            # Optionally, you can process the result here if needed
            pass

    if function and arguments is not None:
        function_calls.append((function, arguments))  # Append the last function call

    return function_calls

# Main route processing API endpoint
@app.post("/process_prompt")
async def process_prompt(request: Prompt):
    global is_monitoring
    start_time = time.time()
    data = request.dict()
    prompt = data['prompt']
    
    llm_invoked = 0  # Initialize llm_invoked to 0 to know whether llm is invoked or not

    try:
        # Start resource monitoring
        is_monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Capture the start time for route selection
        route_start_time = time.time()

        # Get embedding for the prompt
        prompt_embedding, embed_metrics = get_embedding(prompt)

        # Find the best route based on the prompt
        best_route, similarity = find_best_route(prompt_embedding, routes)

        # Calculate the time taken for route selection
        route_selection_duration = (time.time() - route_start_time) * 1e9  # Convert to nanoseconds

        # Check if the similarity score is below the threshold
        if similarity < THRESHOLD:
            best_route = None

        if best_route is None:
            print("No matching route found.")
            route_name = "no_route"
            response = "No route found"
            route_type = "none"
            route_selected = route_name
            # Default values for missing metrics in case no route is found
            prompt_eval_duration = 0
            eval_count = 0
            eval_duration = 0
            prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
            total_duration = embed_metrics.get('total_duration', 0)
            load_duration = embed_metrics.get('load_duration', 0)
            function_execution_time_total = 0
            function_execution_times = []
            llm_response_parsing_duration = 0
            number_of_functions_called = 0
            print(f"No Route Response: {response}")  # Debugging print statement
        else:
            print(f"Selected Route: {best_route.name} with similarity: {similarity}")
            route_name = best_route.name
            route_selected = route_name
            route_type = "dynamic"

            llm_invoked = 1  # Set llm_invoked to 1 since LLM is called
            # For dynamic routes, trigger LLM with multi-shot prompt
            multi_shot_prompt = create_multi_shot_prompt(prompt, best_route)
            llm_response = call_llm_with_multi_shot(multi_shot_prompt, best_route)
            response = llm_response['generated_response']
            # Extract the metrics for dynamic routes
            dynamic_metrics = llm_response['metrics']
            prompt_eval_duration = dynamic_metrics['prompt_eval_duration']
            eval_count = dynamic_metrics['eval_count']
            eval_duration = dynamic_metrics['eval_duration']
            prompt_eval_count = dynamic_metrics['prompt_eval_count']
            total_duration = dynamic_metrics['total_duration']
            load_duration = dynamic_metrics['load_duration']
            function_execution_times = dynamic_metrics.get('function_execution_times', [])
            function_execution_time_total = sum([entry['execution_time_ns'] for entry in function_execution_times])
            llm_response_parsing_duration = dynamic_metrics.get('llm_response_parsing_duration_ns', 0)
            number_of_functions_called = dynamic_metrics.get('number_of_functions_called', 0)
            print(f"Dynamic LLM Response: {response}")  # Debugging print statement

        # Stop resource monitoring
        is_monitoring = False
        monitor_thread.join()

        # Measure resource statistics
        process = psutil.Process()
        memory_usage_before = memory_usage_queue.queue[0] if not memory_usage_queue.empty() else process.memory_info().rss
        memory_usage_after = memory_usage_queue.queue[-1] if not memory_usage_queue.empty() else process.memory_info().rss
        avg_cpu_usage = calculate_average_cpu()
        similarity = round(similarity, 2) if similarity is not None else None

        # Network latency: time spent in network communication for the embedding request
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        total_response_time = (time.time() - start_time) * 1e9  # Convert to nanoseconds

        # Calculate tokens per second for the response
        tokens_per_second = eval_count / eval_duration * 1e9 if eval_duration > 0 else 0
        tokens_per_second = round(tokens_per_second, 2)  # Round to 2 decimal points

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, model_name, embed_model, prompt, response, route_type, route_selected,
            similarity, "cosine", str(prompt_embedding), total_duration, load_duration, prompt_eval_count,
            prompt_eval_duration, eval_count, eval_duration, tokens_per_second,
            avg_cpu_usage, memory_usage_before, memory_usage_after, memory_allocated_for_model,
            network_latency, total_response_time, route_selection_duration, llm_invoked,
            function_execution_time_total, json.dumps(function_execution_times),
            llm_response_parsing_duration, number_of_functions_called
        ]

        # Put the log message into the CSV queue
        print(f"Logging to CSV: {log_message_csv}")  # Debugging print statement
        csv_queue.put(log_message_csv)

        # Return the response to the client
        return {
            "status": "success" if best_route else "no_match",
            "route_selected": route_name,
            "semantic_similarity_score": similarity,
            "similarity_metric": "cosine",
            "response": response
        }

    except Exception as e:
        is_monitoring = False  # Ensure monitoring stops in case of an error
        return {"status": "error", "message": str(e)}

# Function to get embeddings from the embedding model
def get_embedding(text, model=embed_model):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": model, "input": text}
    )
    response_json = response.json()
    return response_json["embeddings"][0], response_json

# Function to calculate average CPU usage during the request
def calculate_average_cpu():
    cpu_usages = []
    while not cpu_usage_queue.empty():
        cpu_usages.append(cpu_usage_queue.get())
    return round(mean(cpu_usages), 2) if cpu_usages else 0

# Function to find the best route based on the prompt
def find_best_route(prompt_embedding, routes):
    best_route = None
    best_similarity = -1

    for route in routes:
        for utterance in route.utterances:
            utterance_embedding, _ = get_embedding(utterance)
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route

    # If best similarity is below the threshold, set best_route to None
    if best_similarity < THRESHOLD:
        best_route = None

    return best_route, best_similarity

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
