import argparse
import os
import openai
from subprocess import call
import sys
import json 

openai.api_key = os.getenv('GPT_API_KEY')
model_engine = "text-davinci-003"
umrf_graph_samples = []

def read_samples(base_path):
		for root, dirs, files in os.walk(base_path):
				path = root.split(os.sep)
				for file in files:
						if not file.endswith(".json"):
								continue

						with open(os.path.join(root, file)) as json_file:
								umrf_graph_samples.append(json.load(json_file))

def generate(prompt_in):
		completion = openai.Completion.create(
				engine=model_engine,
				prompt=prompt_in,
				max_tokens=500,
				n=1,
				stop=None,
				temperature=0.5,
		)

		raw_output = completion.choices[0].text
		print ("RAW GPT OUTPUT:")
		print ("------------------------------------------------------------")
		print (completion)
		print ("------------------------------------------------------------\n")

		return completion.choices[0].text

def text_to_umrf(text_in):
		prompt = "Extract data from natural language sentences (NL_SENTENCE) and store them in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.\n"

		for umrf_graph_sample in umrf_graph_samples:
				example = \
						"NL_SENTENCE: '" + umrf_graph_sample["graph_name"] + "'\n" \
					+ "DESIRED_JSON: " + json.dumps(umrf_graph_sample) + "\n\n"

				prompt = prompt + example

		prompt = prompt + "NL_SENTENCE: '" + text_in + "'\nDESIRED_JSON: "
		print ("FULL PROMPT:")
		print ("------------------------------------------------------------")
		print (prompt)
		print ("------------------------------------------------------------\n")
		return generate(prompt)

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('-ue', '--umrf-examples', required=True, type=str, help='Path for the generated C++ source files')
		parser.add_argument('-is', '--input-sentence', required=True, type=str, help='Input sentence')
		args, unknown = parser.parse_known_args()

		read_samples(args.umrf_examples)
		input_sentence = args.input_sentence

		print ("Extracting UMRF from input: '" + input_sentence + "' ...\n")
		extracted_umrf = text_to_umrf(input_sentence)
		extracted_umrf_json = json.loads(extracted_umrf)

		print ("EXTRACTED UMRF GRAPH: ")
		print ("------------------------------------------------------------")
		print (json.dumps(extracted_umrf_json, indent=2))
		print ("------------------------------------------------------------")
