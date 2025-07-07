import ast
import sys
import time
import json
import argparse
import requests
import traceback
from pathlib import Path
from typing import Dict, Union, List
 
import pandas as pd
from tqdm import tqdm

import anthropic
from tenacity import retry, wait_exponential, stop_after_attempt

from utils import encode_image, get_header

import warnings
warnings.filterwarnings('ignore')


def build_vlm_payload(trial_metadata, task_payload, task=None, api='azure'):
    """
    Parameters:
    trial_metadata (dict): The metadata for the task.
    task (str): The task name.
    task_payload (str): The task payload.

    Returns:
    str: The parsed task prompt.
    """
    # Get rid of old images from the payload.
    if api == 'azure':
        task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]

        # Handle RMTS tasks separately.
        if 'rmts' in task:
            vals = {k: v for k, v in trial_metadata.items() if f'{{{k}}}' in task_payload['messages'][0]['content'][0]['text']}
            task_payload['messages'][0]['content'][0]['text'] = task_payload['messages'][0]['content'][0]['text'].format(**vals)
            condition = task.split('_')[1]
            if condition == 'decomposed':
                img_path = ast.literal_eval(trial_metadata['decomposed_paths'])
                images = [encode_image(path) for path in img_path]
            else:  # unified
                img_path = trial_metadata['unified_path']
                images = [encode_image(img_path)]
        else:
            img_path = trial_metadata['path']
            images = [encode_image(img_path)]

        # Add the image(s) to the payload.
        image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}} for image in images]
        task_payload['messages'][0]['content'] += image_payload

    elif api == 'anthropic':
        task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]
        # Handle RMTS tasks separately.
        if 'rmts' in task:
            vals = {k: v for k, v in trial_metadata.items() if f'{{{k}}}' in task_payload['messages'][0]['content'][0]['text']}
            task_payload['messages'][0]['content'][0]['text'] = task_payload['messages'][0]['content'][0]['text'].format(**vals)
            condition = task.split('_')[1]
            if condition == 'decomposed':
                img_path = ast.literal_eval(trial_metadata['decomposed_paths'])
                images = [encode_image(path) for path in img_path]
            else:  # unified
                img_path = trial_metadata['unified_path']
                images = [encode_image(img_path)]
        else:
            img_path = trial_metadata['path']
            images = [encode_image(img_path)]

        # Add the images to the payload
        image_payload = [{'type': 'image', 'source': {
            'type': 'base64', 'media_type': 'image/png', 'data': image}
        } for image in images]
        task_payload['messages'][0]['content'] += image_payload

    return task_payload


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def run_trial(
    header: Dict[str, str],
    api_metadata: Dict[str, str],
    task_payload: Dict,
    parse_payload: Dict,
    parse_prompt: str,
    api: str
):
    """
    Run a trial of the specified task.

    Parameters:
    header(Dict[str, str]): The API information.
    api_metadata (str): Metadata describing the relevant endpoints for the API request.
    task_payload (Dict): The payload for the vision model request.
    parse_payload (Dict): The payload for the parsing model request.
    parse_prompt (str): The prompt for the parsing model.

    Returns:
    str: The response and the parsed response from the trial.
    """

    # Until the model provides a valid response, keep trying.
    trial_response = requests.post(
        api_metadata['vision_endpoint'],
        headers=header,
        json=task_payload,
        timeout=240
    )

    # Check for easily-avoidable errors
    if 'error' in trial_response.json():
        print('failed VLM request')
        print(trial_response.json())
        raise ValueError('Returned error: \n' + trial_response.json()['error']['message'])
    
    # Extract the responses from the vision model and parse them with the parsing model.
    if api == 'azure':
        trial_response_out = trial_response.json()['choices'][0]['message']['content']
    elif api == 'anthropic':
        trial_response_out = trial_response.json()['content'][0]['text']
    else:
        raise ValueError(f"API not supported: ({api})")
    trial_parse_prompt = parse_prompt + '\n' + trial_response_out
    parse_payload['messages'][0]['content'][0]['text'] = trial_parse_prompt  # update the payload
    answer = requests.post(api_metadata['parse_endpoint'], headers=header, json=parse_payload, timeout=240)
    if api == 'azure':
        parse_tokens = answer.json()['usage']['completion_tokens']
        run_tokens = trial_response.json()['usage']['completion_tokens']
        answer = answer.json()['choices'][0]['message']['content']
    elif api == 'anthropic':
        parse_tokens = answer.json()['usage']['output_tokens']
        run_tokens = trial_response.json()['usage']['output_tokens']
        answer = answer.json()['content'][0]['text']

    # post-json parsing
    if '{' in answer:
        answer = answer.replace('""', '"')

    # If the response is invalid raise an error.
    if 'error' in answer:
        print('failed parsing request')
        raise ValueError('Returned error: \n' + answer['error']['message'])
    elif answer == '-1':
        print('bad VLM response')
        # raise ValueError(f'Invalid response: {trial_response_out}')
    return answer, trial_response_out, run_tokens, parse_tokens


def save_results(results_df: pd.DataFrame, results_file: str=None):
    if results_file:
        results_df.to_csv(results_file, index=False)
    else:
        filename = f'results_{time.time()}.csv'
        results_df.to_csv(filename, index=False)


def parse_args() -> argparse.Namespace:
    '''
    Parse command line arguments.

    Returns:
    argparse.Namespace: The parsed command line arguments.
    '''
    parser = argparse.ArgumentParser(description='Run trials for the specified task.')
    parser.add_argument('--task', type=str, required=True, choices=['search', 'rmts_decomposed', 'rmts_unified', 'counting', 'popout', 'binding'], help='The name of the task.')
    parser.add_argument('--task_dir', type=str, required=True, help='Where the task images and metadata are stored.')
    parser.add_argument('--task_file', type=str, required=True, help='The file containing the task metadata.')
    parser.add_argument('--task_prompt_path', type=str, required=True, help='The location of the prompt file for the task.')
    parser.add_argument('--parse_prompt_path', type=str, required=True, help='The location of the prompt file for parsing the response.')
    parser.add_argument('--results_file', type=str, default=None, help='The file to save the results to.')
    parser.add_argument('--api_file', type=str, default='api_metadata.json', help='Location of the file containing api keys and endpoints.')
    parser.add_argument('--task_payload', type=str, default='payloads/gpt4v_image.json', help='The path to the task payload JSON file.')
    parser.add_argument('--parse_payload', type=str, default='payloads/gpt4_parse.json', help='The prompt for parsing the response.')
    parser.add_argument('--max_vision_tokens', type=int, default=500, help='The maximum number of tokens for the VLM API request.')
    parser.add_argument('--max_parse_tokens', type=int, default=5, help='The maximum number of tokens for the response parsing API request.')
    parser.add_argument('--n_trials', type=int, default=None, help='The number of trials to run. Leave blank to run all trials.')
    parser.add_argument('--api', type=str, default='azure', help='Which API to use for the requests.')
    parser.add_argument('--log_interval', type=int, default=10, help='The interval at which to save the results.')
    parser.add_argument('--sleep', type=int, default=0, help='The time to sleep between requests.')
    parser.add_argument('--replace_results', action='store_true', help='Replace the results file if it already exists.')
    parser.add_argument('--dynamic-token', action='store_true', help='Use dynamic token allocation.')
    return parser.parse_args()


def main():
    # Parse command line arguments.
    args = parse_args()
    print('Running trials for task:', args.task)

    # Automatically select the payload files based on the API.
    if args.api == 'azure':
        args.task_payload = 'payloads/gpt4v_image.json'
        args.parse_payload = 'payloads/gpt4_parse.json'
    elif args.api == 'anthropic':
        args.task_payload = 'payloads/claude_image.json'
        args.parse_payload = 'payloads/claude_parse.json'

    # Load the relevant payloads and prompts.
    task_payload = json.load(open(args.task_payload, 'r'))
    parse_payload = json.load(open(args.parse_payload, 'r'))
    api_metadata = json.load(open(args.api_file, 'r'))
    parse_prompt = Path(args.parse_prompt_path).read_text()
    task_prompt = Path(args.task_prompt_path).read_text()
    # task_payload['messages'][0]['content'][0]['text'] = task_prompt
    task_payload['max_tokens'] = args.max_vision_tokens
    parse_payload['max_tokens'] = args.max_parse_tokens

    # OpenAI API Key and header.
    header = get_header(api_metadata, model=args.api)
    api_metadata = api_metadata[args.api]

    # Load the task metadata and results.
    try:
        results_df = pd.read_csv(args.results_file)
        print(f"Loaded results from {args.results_file}")
        if 'response' not in results_df.columns or 'answer' not in results_df.columns or args.replace_results:
            print(f"Replacing results: {'response' not in results_df.columns}, {'answer' not in results_df.columns}, {args.replace_results}")
            response_df = pd.DataFrame(columns=['response', 'answer'], dtype=str)
            response_df[['response', 'answer']] = ''
            results_df = pd.concat([response_df, results_df], axis=1)
    except (FileNotFoundError, ValueError):
        # If no valid results_df was provided, open the task metadata and construct a new one.
        metadata_df = pd.read_csv(args.task_file)
        print(f"Loaded metadata from {args.task_file}")
        if 'response' not in metadata_df.columns or 'answer' not in metadata_df.columns or args.replace_results:
            # drop response if it exists, do nothing if it doesn't
            if 'response' in metadata_df.columns:
                metadata_df = metadata_df.drop(columns=['response'])
            results_df = pd.DataFrame(columns=['response', 'answer'], dtype=str)
            results_df[['response', 'answer']] = ''
            results_df = pd.concat([metadata_df, results_df], axis=1)
        else:
            results_df = metadata_df

    # Shuffle the trials, extracting n_trials if the argument was specified
    if args.n_trials:
        results_df = results_df.sample(n=args.n_trials).reset_index(drop=True)
    else:
        results_df = results_df.sample(frac=1).reset_index(drop=True)

    # Run all the trials.
    p_bar = tqdm(total=len(results_df))
    parse_token_max, run_token_max = 0, 0
    trials_left = len(results_df[results_df.apply(lambda x: str(x['response']) == "nan", axis=1)])
    for i, trial in results_df.iterrows():
        # Only run the trial if it hasn't been run before.
        if type(trial.response) != str or trial.response == '0.0':
            print(f"Trials left: {trials_left}/{len(results_df)}")
            print(f"Running trial {i} for img_id {trial.img_id}...")
            try:
                # deepcopy the task payload before passing it to the function.
                trial_payload = task_payload.copy()
                trial_payload['messages'][0]['content'][0]['text'] = task_prompt
                task_payload = build_vlm_payload(trial, trial_payload, task=args.task, api=args.api)

                answer, trial_response, run_tokens, parse_tokens = run_trial(
                    header, api_metadata, task_payload, parse_payload, parse_prompt, api=args.api)

                # update the number of max tokens to be 1.2 * the number of max tokens seen
                if args.dynamic_token:
                    if i < 10:
                        parse_token_max = max(parse_token_max, parse_tokens)
                        run_token_max = max(run_token_max, run_tokens)
                    else:
                        parse_token_max = max(parse_token_max, parse_tokens)
                        run_token_max = max(run_token_max, run_tokens)
                        parse_payload['max_tokens'] = max(3, int(parse_token_max) + 1)
                        task_payload['max_tokens'] = max(10, int(1.1 * run_token_max))

                # progress bar display
                if '{' in answer:
                    p_bar_answer = 'json'
                else:
                    p_bar_answer = answer
                p_bar.set_postfix({'answer': p_bar_answer, 'run_tokens': run_tokens, 'parse_tokens': parse_tokens})

                results_df.loc[i, 'response'] = trial_response
                results_df.loc[i, 'answer'] = answer
                time.sleep(args.sleep)
            except Exception as e:
                print(f'Failed on trial {i} with error: {e}')
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                break  # Stop the loop if there is an error and save the progress.
            trials_left-=1
        else:
            print(f"Skipping trial {i} for img_id {trial.img_id}...")
        # Save the progress at log_interval.
        if i % args.log_interval == 0:
            save_results(results_df, args.results_file)
        p_bar.update()
    p_bar.close()

    # Save the final results.
    save_results(results_df, args.results_file)


if __name__ == '__main__':
    main()