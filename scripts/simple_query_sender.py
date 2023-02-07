import argparse
import os
import openai
from subprocess import call
import sys

openai.api_key = os.getenv('GPT_API_KEY')
model_engine = "text-davinci-003"

def generate(prompt_in):
		completion = openai.Completion.create(
				engine=model_engine,
				prompt=prompt_in,
				max_tokens=2048,
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

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('-pf', '--prompt-file', required=True, type=str, help='Input sentence')
		args, unknown = parser.parse_known_args()

		contents = ""
		with open(args.prompt_file, "r") as f:
				contents = f.read()
    
		print (contents)
		generate(contents)
