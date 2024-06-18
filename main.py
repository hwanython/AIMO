import os
import time
import torch
import gc
from numpy.random import choice
import numpy as np
torch.backends.cuda.enable_mem_efficient_sdp(False)
import pandas as pd
from tqdm import tqdm
import re
import math
import random
from collections import defaultdict
from collections import Counter

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed
)
from utils.test_utils import (naive_parse,
                              return_last_print, 
                              process_code, 
                              process_text_output
)
import argparse
import yaml
import tqdm
import logging
from munch import munchify


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop,last_token)):
                return True
        return False
        
if __name__ == "__main__":

    
    LOGGER=True

    if LOGGER:
        log_fn = 'log.txt'
        if os.path.exists(log_fn):
            os.remove(log_fn)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Create a file handler
        file_handler = logging.FileHandler(log_fn)
        file_handler.setLevel(logging.INFO)
        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="./configs/config.yaml",
                            help="the config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")

    args = arg_parser.parse_args()
    
    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit
    
    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r", encoding='utf-8'), yaml.FullLoader, )
    config = munchify(config)
    

    import aimo
    env = aimo.make_env()
    iter_test = env.iter_test()


    import transformers
    logging.info(f"Transformers Version: {transformers.__version__}")
    SEED = config.SEED
    set_seed(SEED)

    NOTEBOOK_START_TIME = time.time()
    DEBUG = config.DEBUG
    QUANT = config.QUANT
    USE_PAST_KEY = config.USE_PAST_KEY
    PRIVATE = config.PRIVATE

    if QUANT:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    torch.cuda.empty_cache()
    gc.collect()




    
    n_repetitions = config.n_repetitions # Original notebook had 22 but times out :(
    TOTAL_TOKENS = config.TOTAL_TOKENS # if PRIVATE else 512

    if PRIVATE:
        TIME_LIMIT = 31500
    else:
        TIME_LIMIT = 31500 # ORIGIN 1



    if PRIVATE:

        MODEL_PATH = "./deepseek-math" #"/kaggle/input/gemma/transformers/7b-it/1"
        DEEP = config.DEEP

        config = AutoConfig.from_pretrained(MODEL_PATH)
        config.gradient_checkpointing = True

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        device_map = [('model.embed_tokens', 0),
                    ('model.layers.0', 0),
                    ('model.layers.1', 0),
                    ('model.layers.2', 0),
                    ('model.layers.3', 0),
                    ('model.layers.4', 0),
                    ('model.layers.5', 0),
                    ('model.layers.6', 0),
                    ('model.layers.7', 0),
                    ('model.layers.8', 0),
                    ('model.layers.9', 0),
                    ('model.layers.10', 0),
                    ('model.layers.11', 0),
                    ('model.layers.12', 0),
                    ('model.layers.13', 0),
                    ('model.layers.14', 0),
                    ('model.layers.15', 0),
                    ('model.layers.16', 0),
                    ('model.layers.17', 0),
                    ('model.layers.18', 0),
                    ('model.layers.19', 0),
                    ('model.layers.20', 0),
                    ('model.layers.21', 0),
                    ('model.layers.22', 1),
                    ('model.layers.23', 1),
                    ('model.layers.24', 1),
                    ('model.layers.25', 1),
                    ('model.layers.26', 1),
                    ('model.layers.27', 1),
                    ('model.layers.28', 1),
                    ('model.layers.29', 1),
                    ('model.norm', 1),
                    ('lm_head', 1)]

        device_map = {ii:jj for (ii,jj) in device_map}

        if QUANT:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="sequential",
                torch_dtype="auto",
                trust_remote_code=True, 
                quantization_config=quantization_config,
                config=config
            )
        else:  
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map=device_map,
                torch_dtype="auto",
                trust_remote_code=True,
                #quantization_config=quantization_config,
                config=config
            )
        
        pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype='auto',
        device_map=device_map,
    )



        stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output"] #,  
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        logging.info(model.dtype, model.hf_device_map)


    code = config.code


    cot = config.cot
    promplt_options = [code,cot]

    tool_instruction = config.tool_instruction


    temperature = config.temperature
    top_p = config.top_p

    temperature_coding = temperature
    top_p_coding = top_p

    
    total_results = {}
    total_answers = {}
    best_stats = {}
    total_outputs = {}
    question_type_counts = {}
    starting_counts = (2,3)
        
    # LEWIS: I had to invert the loop order because the new API forbids repeated calls on the same problem
    for i, (test, sample_submission) in tqdm(enumerate(iter_test)):
        logging.info(f"Solving problem {i} ...")
        TIME_SPENT = time.time() - NOTEBOOK_START_TIME

        if TIME_SPENT>TIME_LIMIT:
            break
            
        for jj in tqdm(range(n_repetitions)):   
    #     for i, (test, sample_submission) in tqdm(enumerate(iter_test)):
            

    #         id_ = df['id'].loc[i]
    #         problem = df['problem'].loc[i]
            problem = test['problem'].values[0]
            logging.info(f"\n\n\nQUESTION {i} - {jj} - TIME_SPENT : {TIME_SPENT:.0f} secs")
            
            best, best_count = best_stats.get(i,(-1,-1))
            if best_count>np.sqrt(jj):
                logging.info("SKIPPING CAUSE ALREADY FOUND BEST")
                continue
                
            outputs = total_outputs.get(i,[])
            text_answers, code_answers = question_type_counts.get(i,starting_counts)
            results = total_results.get(i,[])
            answers = total_answers.get(i,[])
            
            for _ in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.2)
            # Main loop
            try:
                ALREADY_GEN = 0
                code_error = None
                code_error_count = 0
                code_output = -1
                #initail_message = problem  + tool_instruction 
                counts = np.array([text_answers,code_answers])

                draw = choice(promplt_options, 1,
                            p=counts/counts.sum())

                initail_message = draw[0].format(problem,"{}")            
                prompt = f"User: {initail_message}"

                current_printed = len(prompt)
                logging.info(f"{jj}_{prompt}\n")

                model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device) # return_tensor as pytorch tensor
                input_len = len(model_inputs['input_ids'][0])

                generation_output = model.generate(**model_inputs, 
                                                max_new_tokens=TOTAL_TOKENS-ALREADY_GEN,
                                                return_dict_in_generate=USE_PAST_KEY,
                                                do_sample = True,
                                                temperature = temperature,
                                                top_p = top_p,
                                                num_return_sequences=1, stopping_criteria = stopping_criteria)

                if USE_PAST_KEY:
                    output_ids = generation_output.sequences[0]
                else:
                    output_ids = generation_output[0]
                decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                logging.info(f"{decoded_output[current_printed:]}\n")
                current_printed += len(decoded_output[current_printed:])
                cummulative_code = ""
                
                
                stop_word_cond = False
                for stop_word in stop_words:
                    stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)
                    
                
                while (stop_word_cond) and (ALREADY_GEN<(TOTAL_TOKENS)):

                    if (decoded_output[-len("```python"):]=="```python"):
                        temperature_inner=temperature_coding
                        top_p_inner = top_p_coding
                        prompt = decoded_output
                    else:
                        temperature_inner=temperature
                        top_p_inner = top_p
                        try:
                            if (decoded_output[-len("``````output"):]=="``````output"):
                                code_text = decoded_output.split('```python')[-1].split("``````")[0]
                            else:
                                code_text = decoded_output.split('```python')[-1].split("```")[0]
                            

                            cummulative_code+=code_text
                            code_output, CODE_STATUS = process_code(cummulative_code, return_shell_output=True)
                            logging.info(f'CODE RESULTS: {code_output}', )

                            if code_error==code_output:
                                code_error_count+=1
                            else:
                                code_error=code_output
                                code_error_count = 0

                            if not CODE_STATUS:
                                cummulative_code = cummulative_code[:-len(code_text)]

                                if code_error_count>=1:
                                    logging.info("REPEATED ERRORS")
                                    break

                        except Exception as e:
                            logging.error(e)
                            logging.error('ERROR PARSING CODE')
                            code_output = -1

                        if code_output!=-1:
                            if (decoded_output[-len(")\n```"):]==")\n```"):
                                prompt = decoded_output+'```output\n'+str(code_output)+'\n```\n'
                            else:
                                prompt = decoded_output+'\n'+str(code_output)+'\n```\n'
                        else:
                            prompt = decoded_output
                            cummulative_code=""


                    model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                    ALREADY_GEN =  len(model_inputs['input_ids'][0])-input_len

                    if USE_PAST_KEY:
                        old_values = generation_output.past_key_values
                    else:
                        old_values = None

                    generation_output = model.generate(**model_inputs, 
                                                    max_new_tokens=TOTAL_TOKENS-ALREADY_GEN, 
                                                    return_dict_in_generate=USE_PAST_KEY,
                                                    past_key_values=old_values,
                                                    do_sample = True,
                                                    temperature = temperature_inner,
                                                    top_p = top_p_inner,
                                                    num_return_sequences=1, stopping_criteria = stopping_criteria)

                    if USE_PAST_KEY:
                        output_ids = generation_output.sequences[0]
                    else:
                        output_ids = generation_output[0]
                    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                    logging.info(f"\nINTERMEDIATE OUT :\n{decoded_output[current_printed:]}\n")
                    current_printed+=len(decoded_output[current_printed:])
                    
                    stop_word_cond = False
                    for stop_word in stop_words:
                        stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)

                if USE_PAST_KEY:
                    output_ids = generation_output.sequences[0]
                else:
                    output_ids = generation_output[0]

                raw_output = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
                #print(f"\n\nOutput :\n{raw_output}\n")                            
                result_output = process_text_output(raw_output)
                
                try:
                    code_output = round(float(eval(code_output))) % 1000
                except Exception as e:
                    logging.error(f"{e} final_eval")
                    code_output = -1

            except Exception as e:
                logging.error(f"{e} 5")#TODO: ERROR HANDLING
                result_output, code_output = -1, -1

            if code_output!=-1:
                outputs.append(code_output)
                code_answers+=1

            if result_output!=-1:
                outputs.append(result_output)
                text_answers+=1

            if len(outputs) > 0:
                occurances = Counter(outputs).most_common()
                logging.info(occurances)
                if occurances[0][1] > best_count:
                    logging.info("GOOD ANSWER UPDATED!")
                    best = occurances[0][0]
                    best_count = occurances[0][1]
                if occurances[0][1] > 5:
                    logging.info("ANSWER FOUND!")
                    break

            results.append(result_output)
            answers.append(code_output)
            
            best_stats[i] = (best, best_count) 
            question_type_counts[i] = (text_answers, code_answers)
            total_outputs[i] = outputs
            
            total_results[i] = results
            total_answers[i] = answers

            logging.info(f"code_answers{code_answers-starting_counts[1]} text_answers {text_answers-starting_counts[0]}")
            if DEBUG:
                break
                
        logging.info(f"Predicted best answer: {best_stats}")
        sample_submission['answer'] = best_stats[i][0]
        env.predict(sample_submission)