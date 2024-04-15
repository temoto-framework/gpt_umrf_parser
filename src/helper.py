"""
This section handles all the configs:
model_type: {llama-7b, codellama-7b, mistral, mixtral, gpt-3}
task_instruction_template_type: {recommended_model_type, verbose, minimal}
test_set: {id_test_set, ood_domain_test_set, ood_task_type_set}
"""


def get_instruction_template(instruct_type: str) -> str:
    s = ""
    if instruct_type == "rp":
        s = ("You are a robot planner tasked with taking the natural language instructions and coordinates "
             " and transpiling them in a JSON format. I will provide you examples of the desired JSON structure.\n")
    elif instruct_type == "minimal":
        s = ("Please translate the data from natural language sentences (NL_SENTENCE) to  "
             "a JSON format. I will provide you examples of the desired JSON structure.\n")
    elif instruct_type == "llama-7b":
        s = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
             " while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
             "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
             "and positive in nature. If a question does not make any sense, or is not factually coherent, "
             "explain why instead of answering something not correct. If you don't know the answer "
             "to a question, please don't share false information.\n")
    elif instruct_type == "codellama":
        s = ("You are an expert programmer that writes simple, concise code and explanations. Write translations "
             "of natural language instructions into JSON code.\n")
    elif instruct_type == "mistral" or instruct_type == "mixtral":
        s = ("You are a helpful code assistant. Your task is to generate a valid JSON object based on the given "
             "information:"
             "Just generate the JSON object without explanations:\n")
    elif instruct_type == "gpt3":
        s = "Translate the following sentence into a JSON task graph:\n"
    return s
