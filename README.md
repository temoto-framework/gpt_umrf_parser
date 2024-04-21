# gpt_umrf_parser
GPT-based natural language to UMRF parser. This package contains the code for replicating the code in the journal paper:

Unlocking Underrepresented Use-Cases for Large Language Model-Driven Human-Robot Task Planning.

The repository is broken into three main branches:
* main - provides quick and simple scripts for querying OpenAI API with natural language for UMRF task and motion planning.
* selma-devel - provides the datasets, LLM code for huggingface API and OpenAI API, and prompt builder to perform the prompt experiments in Section 3.
* demo_test - provides the ROS code to perform the remote inspection demonstration in Section 5.

Please refer to each branch's README.md file for full setup instructions.

## Main Branch Setup
* OpenAI API requires Numpy `v1.20.3` or higher, so if that's not the case then: `sudo pip install numpy --upgrade` 
* Install openai python package: `pip install openai`.
* [Generate a key](https://beta.openai.com/account/api-keys) for using openai API.
* Store this key in a separate file.
* Clone this project `git clone https://github.com/temoto-framework/gpt_umrf_parser`

## Usage

### Standalone

``` bash
# Export the key as an environment variable
export GPT_API_KEY=$(cat <path/to/openai_key>)

# Invoke the script
python scripts/gpt_umrf_parser_standalone.py -ue umrf_examples/ -is "Robot go scan the lab [x=111.2; y=87.6; yaw=-0.11]."
```

### ROS node

``` bash
# Export the key as an environment variable
export GPT_API_KEY=$(cat <path/to/openai_key>)

# Invoke the node
rosrun gpt_umrf_parser gpt_umrf_parser_node.py -ue umrf_examples/

# Publish the command
rostopic pub /command std_msgs/String "data: 'Robot go scan the lab [x=111.2; y=87.6; yaw=-0.11].'"

# Subscribe to the result
rostopic echo /broadcast_start_umrf_graph
```
