
class JiangPrompt():
    def __init__(self, input_information: tuple[str, str, str]):
        self.input_information = input_information
    
    """
    Calculates the character-level accuracy for a single
    UMRF ground-truth against a single UMRF model decoding
    """
    def acc(self, model_output:str):
        ground_truth = self.input_information[2].lower()
        model_output = model_output.lower()

        penalize_extra_decodings = len(model_output) - len(ground_truth)

        num_correct_characters = 0
        i = 0
        if penalize_extra_decodings >= 0:
            for char in ground_truth:
                if char == model_output[i]:
                    num_correct_characters = num_correct_characters + 1
                i = i + 1
            
            acc = num_correct_characters/(len(ground_truth) + penalize_extra_decodings)

        else:
            for char in model_output:
                if char == ground_truth[i]:
                    num_correct_characters = num_correct_characters + 1
                i = i + 1
            
            acc = num_correct_characters/(len(model_output) + penalize_extra_decodings)

        return acc

        
        
    
