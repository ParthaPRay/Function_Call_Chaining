# Function_Call_Chaining
This repo shows the codes for Function Call Chaining involving LLMs in Raspberry Pi 4B.

Two Approaches of function call chaining involvng localized quantized LLMs have been used in Raspberry PI 4B with FastAPI + Ollama frameworks:

Function chaining is performed with various functins involving remote API calls and local system and local external function calls. Output of one funtion is considered as input to next funtion and so on. 

1. Make a virtual environment.
2. First run python3 chain_function_call_XY.py in one terminal under the virtual envirnment.
3. Run then curl_caller_configuration_XY.py in other terminal (with / without virtual environment). cURL should be used for such RESTful API call.

Here, **XY** denotes the configuration such as 44, 53, 61, 71 for repeatative study varying LLMs.   It has not special menaing but for development of these codes during my study.


