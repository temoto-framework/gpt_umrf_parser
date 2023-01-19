import os
import openai
import json

class GptConfig:
		def __init__ (self, api_key, engine, logprobs, max_tokens):
				self.api_key = api_key
				self.engine = engine
				self.logprobs = logprobs
				self.max_tokens = max_tokens
				
class GptUmrfParser:
		def __init__(self, gpt_config, examples):
				self.gpt_config = gpt_config
				openai.api_key = self.gpt_config.api_key
				self.examples = examples

		def generate(self, prompt_in):
				completion = openai.Completion.create(
						engine = self.gpt_config.engine,
						max_tokens = self.gpt_config.max_tokens,
						logprobs = self.gpt_config.logprobs,
						temperature = 0.5,
						n = 1,
						stop = None,
						prompt = prompt_in
				)

				output_text = completion.choices[0].text
				output_raw = completion
				return output_text, output_raw

		def text_to_umrf(self, text_in):
				prompt = "Extract data from natural language sentences (NL_SENTENCE) and store them in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.\n"

				for example in self.examples:
						formated_example = \
								"NL_SENTENCE: '" + example[0] + "'\n" \
							+ "DESIRED_JSON: " + example[1] + "\n\n"

						prompt = prompt + formated_example

				prompt = prompt + "NL_SENTENCE: '" + text_in + "'\nDESIRED_JSON: "

				print ("FULL PROMPT:")
				print ("------------------------------------------------------------")
				print (prompt)
				print ("------------------------------------------------------------\n")

				return self.generate(prompt)

def read_sample_jsons(base_path):
		umrf_graph_samples = []
		for root, dirs, files in os.walk(base_path):
				path = root.split(os.sep)
				for file in files:
						if not file.endswith(".json"):
								continue

						with open(os.path.join(root, file)) as json_file:
								umrf_sample = json.load(json_file)
								umrf_graph_samples.append((umrf_sample["graph_name"], json.dumps(umrf_sample)))

		return umrf_graph_samples

def to_pretty_json_str(umrf_json_str):
		umrf_json = json.loads(umrf_json_str)
		return json.dumps(umrf_json, indent=2)
