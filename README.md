# gpt_umrf_parser
GPT-based natural language to UMRF parser

## Setup
* OpenAI API requires Numpy `v1.20.3` or higher, so if that's not the case then: `sudo pip install numpy --upgrade` 
* Install openai python package: `pip install openai`.
* [Generate a key](https://beta.openai.com/account/api-keys) for using openai API.
* Store this key in a separate file.
* Clone this project `git clone https://github.com/temoto-framework/gpt_umrf_parser`

## Usage

``` bash
# Export the key as an environment variable
export GPT_API_KEY=$(cat <path/to/openai_key>)

# Invoke the script
python scripts/gpt_umrf_parser.py -ue umrf_examples/ -is "Drive close to the fire"
```

Expected output:
```
Extracting UMRF from input: 'Drive close to the fire' ...

FULL PROMPT:
------------------------------------------------------------
Extract data from natural language sentences (NL_SENTENCE) and store them in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.
NL_SENTENCE: 'Go to the workshop'
DESIRED_JSON: {"graph_name": "Go to the workshop", "umrf_actions": [{"name": "navigation", "description": "Go to the workshop.", "id": 0, "input_parameters": {"location": {"label": {"pvf_type": "string", "pvf_value": "workshop"}}}}]}

NL_SENTENCE: 'Move to the main hall.'
DESIRED_JSON: {"graph_name": "Move to the main hall.", "umrf_actions": [{"name": "navigation", "description": "Move to the main hall.", "id": 0, "input_parameters": {"location": {"label": {"pvf_type": "string", "pvf_value": "main hall"}}}}]}

NL_SENTENCE: 'Drive close to the fire'
DESIRED_JSON: 
------------------------------------------------------------

RAW GPT OUTPUT:
------------------------------------------------------------
 {"graph_name": "Drive close to the fire", "umrf_actions": [{"name": "navigation", "description": "Drive close to the fire.", "id": 0, "input_parameters": {"location": {"label": {"pvf_type": "string", "pvf_value": "fire"}}}}]}
------------------------------------------------------------

EXTRACTED UMRF GRAPH: 
------------------------------------------------------------
{
  "graph_name": "Drive close to the fire",
  "umrf_actions": [
    {
      "name": "navigation",
      "description": "Drive close to the fire.",
      "id": 0,
      "input_parameters": {
        "location": {
          "label": {
            "pvf_type": "string",
            "pvf_value": "fire"
          }
        }
      }
    }
  ]
}
------------------------------------------------------------
```