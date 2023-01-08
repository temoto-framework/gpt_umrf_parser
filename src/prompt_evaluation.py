# TODO:

"""
This script will measure the BLEU score variances as commentary on 
the UMRF task's prompt engineering robustness.

Method 1. Greedy Search: Measure BLEU score variance along the axes of
+ number of few-shot examples included in the prompt (1, 2, or 3 examples)
+ ordering of few-shot examples of same length in the prompt
+ selection of few-shot examples of same length in the prompt

+ report the optimal prompt design w/ BLEU score
+ provide commentary on LLM difficulty to generate long sequences, i.e.,
  UMRF graph decoding w/ a single node success vs UMRF graph decoding with more than one node

Method 2. Reinforcement Learning:
+ which augmetnation-actions provide the best improvement on BLEU acc overall
+ which augmentation-actions hurt performance in general

+ report the optimal prompt desing w/ BLEU score
"""