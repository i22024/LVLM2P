#!/usr/bin/env python3
import logging
import time
import numpy as np
from matplotlib import pyplot as plt

from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import re

from gym_minigrid.wrappers import *
from image_preprocess import CUSTOMIZED_ACTION_VOCAB, INTERACTION_ACTIONS, create_image_from_2_sequences, create_image_from_1_sequence, create_single_image_without_wall

from prompt_gemini_2turn_coord import fewshot_prompt_coord
from prompt_gemini_2turn_32shot import fewshot_prompt_action

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import GenerationConfig
from google.generativeai.types.generation_types import BlockedPromptException, BrokenResponseError


gemini_api_keys = []

logger = logging.getLogger(__name__)


def my_print(text, log=True):
    if log:
        logging.info(text)

gemini_api_keys = np.random.permutation(gemini_api_keys).tolist()
gemini_api_keys_for_mp = []
global_key_idx_for_mp = []

def construct_gemini_keys(num_rollout_workers=1):
    # For multiprocess, divide keys across processes to avoid duplicate usage
    global gemini_api_keys_for_mp
    global global_key_idx_for_mp

    cnt = 0
    gemini_api_keys_for_mp = []
    for i in range(num_rollout_workers):
        gemini_api_keys_for_mp.append([gemini_api_keys[cnt]])
        cnt += 1
        if cnt >= len(gemini_api_keys):
            cnt = 0
    for i, key in enumerate(gemini_api_keys[num_rollout_workers:]):
        rank = i % num_rollout_workers
        gemini_api_keys_for_mp[rank].append(key)

    global_key_idx_for_mp = [0] * num_rollout_workers
    


def construct_vlm_with_key(rank=0):
    # Set up the model parameters: https://ai.google.dev/gemini-api/docs/models/generative-models#model_parameters
    # https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pro-config-example
    global gemini_api_keys_for_mp
    global global_key_idx_for_mp
    gemini_key = gemini_api_keys_for_mp[rank][global_key_idx_for_mp[rank]]
    genai.configure(api_key=gemini_key)
   



    #### Call gemini 1.5 flash
    gemini_flash = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=GenerationConfig(
            temperature=0, top_p=0.95, top_k=0, max_output_tokens=4096,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
   
    #### Call gemini 1.5 Pro
    gemini_pro = genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0801", # gemini-1.5-pro-exp-0801
        generation_config=GenerationConfig(
            temperature=0, top_p=0.95, top_k=0, max_output_tokens=4096,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )



    global_key_idx_for_mp[rank] += 1
    if global_key_idx_for_mp[rank] == len(gemini_api_keys_for_mp[rank]):
        global_key_idx_for_mp[rank] = 0
        
    return gemini_flash, gemini_pro , gemini_key

def query_probability_from_vlm_with_images(
        gemini_flash,gemini_pro, image,  target_object, save_idx,
        include_start_state=True, query_for_preference=False, dense_reward=False, sleep=0,
):

    error = None
    # =========================================== BEGINNING OF FIRST PROMPT SCOPE ===========================================
    """
    ##################################################################################################
    #                     Prompt for two-turn query with single image                                #
    ##################################################################################################
    """

    # Get coordinates and relative position of objects
    prompt_coordinate=(
        f"In this 6x6 grid image, in which row and column is the {target_object} based on the top left?"
        f"In which row and column is the orange arrow located, and in which direction is it facing?"
        f"Tip: The numbers written vertically between the {target_object} and orange arrow are rows, and the numbers written horizontally are columns."
        f"On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, {target_object}'s row, {target_object}'s column, 'orange arrow's direction']."
    )
    # ============================================== END OF FIRST PROMPT SCOPE ==============================================

    gemini_log = ""
    chat_gemini_coordinate = gemini_flash.start_chat(history=fewshot_prompt_coord) 

    try:
        responses = chat_gemini_coordinate.send_message([image, prompt_coordinate], stream=True)
        responses.resolve()
    except Exception as error:
        print(f"[INFO] Error Occured when making VLM response: {error}")
        return None, [], error

    gemini_coordinate = responses.text
    gemini_coordinate = gemini_coordinate.strip()
    print(gemini_coordinate)
    time.sleep(0.4)
    gemini_log += str(gemini_coordinate) + "\n"
    gemini_log += "------------------------------------ Turn --------------------------------"
    #reward_answer = gemini_analysis.replace('\n', '').replace('[', '').replace(']', '').replace('"', '')  # Remove redundant character
    match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*'(\w+)'\]", gemini_coordinate)
    coord_list_result=[0,0,0,0,0,0]
    if match:
        coord_list_result = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), match.group(5)]
        print(coord_list_result)
    else:
        print("No match found")
    

    target_object_row=int(coord_list_result[2])
    target_object_column=int(coord_list_result[3])
    agent_row=int(coord_list_result[0])
    agent_column=int(coord_list_result[1])
    agent_direction=str(coord_list_result[4])

    # calculate first target position
    if target_object_row==1:
        target_location_row=target_object_row+1
        target_location_column=target_object_column
    elif target_object_row==6:
        target_location_row=target_object_row-1
        target_location_column=target_object_column
    elif target_object_column==1:
        target_location_row=target_object_row
        target_location_column=target_object_column+1
    elif target_object_column==6:
        target_location_row=target_object_row
        target_location_column=target_object_column-1
    else:
        target_location_row=target_object_row
        target_location_column=target_object_column
        print("target_object's coordinate wrong!")

    related_position="upwards"
    # when agent is on the one step before : target is door
    if agent_row==target_location_row and agent_column==target_location_column :
        if agent_row==target_object_row and agent_column<target_object_column:
            related_position="rightwards"
        elif agent_row==target_object_row and agent_column>target_object_column:
            related_position="leftwards"
        elif agent_row>target_object_row and agent_column==target_object_column:
            related_position="upwards"    
        elif agent_row<target_object_row and agent_column==target_object_column:
            related_position="downwards"  
        
        elif agent_row>target_object_row and agent_column<target_object_column:
            related_position="rightwards and upwards" 
        
        elif agent_row<target_object_row and agent_column<target_object_column:
            related_position="rightwards and downwards" 

        elif agent_row>target_object_row and agent_column>target_object_column:
            related_position="leftwards and upwards" 

        elif agent_row<target_object_row and agent_column>target_object_column:
            related_position="leftwards and downwards" 
    # when agent is not on the one step before : target is one step before
    else:
        if agent_row==target_location_row and agent_column<target_location_column:
            related_position="rightwards"
        elif agent_row==target_location_row and agent_column>target_location_column:
            related_position="leftwards"
        elif agent_row>target_location_row and agent_column==target_location_column:
            related_position="upwards"    
        elif agent_row<target_location_row and agent_column==target_location_column:
            related_position="downwards"  
        elif agent_row>target_location_row and agent_column<target_location_column:
            related_position="rightwards and upwards" 
        
        elif agent_row<target_location_row and agent_column<target_location_column:
            related_position="rightwards and downwards" 

        elif agent_row>target_location_row and agent_column>target_location_column:
            related_position="leftwards and upwards" 

        elif agent_row<target_location_row and agent_column>target_location_column:
            related_position="leftwards and downwards" 
    # =========================================== BEGINNING OF SECOND PROMPT SCOPE ===========================================
    # Get Probability Distribution of Gemini Policy
    # Orange arrow should not face towards row 1, row 6, column 1, or column 6, except for {target_object}.
    prompt_32shot=f"""
            Position relationship:
            Orange arrow is facing {agent_direction}.
            Target location is to the {related_position} from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
    """

    prompt_distribution=f"""
    You must follow these Orange arrow rules:
    'Go forward' moves forward one space in the direction the orange arrow is pointing.
    If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
    If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
    If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
    If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
    If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
    If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
    If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
    If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.
    
    Current State:
    Based on top left (1,1),
    Orange arrow's current location : row {agent_row} , column {agent_column}
    Orange arrow's current pointing : {agent_direction}
    Current target location : row {target_location_row}, column {target_location_column}

    Problem :  Is the orange arrow currently at the target location?
    If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row {target_object_row} column {target_object_column}.
    If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row {target_location_row} column {target_location_column}.
    On the last line, Reply your answer as a single list.
    """
    # ============================================== END OF SECOND PROMPT SCOPE ==============================================

    chat_gemini_prob = gemini_flash.start_chat(history=fewshot_prompt_action) 

    try:
        responses = chat_gemini_prob.send_message([prompt_32shot], stream=True)
        responses.resolve()
    except Exception as error:
        print(f"[INFO] Error Occured when making VLM response: {error}")
        return None, [], error

    gemini_action_probability = responses.text
    gemini_action_probability = gemini_action_probability.strip()
    pattern = r"\[([0-9.,\s]+)\]"
    match = re.search(pattern, gemini_action_probability)
    if match:
        number_strings = match.group(1).split(',')
        prompt_distribution= [float(num.strip()) for num in number_strings]
    else:
        print("No match found")
        breakpoint()
    
    gemini_log += str(gemini_action_probability) + "\n"

    return prompt_distribution, gemini_log, error

def get_reward_with_vlm(
        raw_observations, actions, instruction,data_idx, obs_size=256,
        at_last_step=False, include_start_state=True, default_window_size=5, query_for_preference=False,
        dense_reward=False, threshold_score=10,
        rank=0, sleep=0,
):

    ""
    """
    Important parameters:
        @at_last_step: query the VLM for last time step of the sequence in raw_observations, True is used during
            training RL, False is used during testing reward module in full trajectories
        @include_start_state: If True, then the image used to ask VLM contains 'window_size + 1' observations including
            the start observation, and 'window_size' actions in the prompt. If False, then the image contains
            'window_size' observations, and 'window_size' actions in the prompt.
        @query_for_preference: If True, then we query VLM for preference of two sequences, False is query for one sequence.
        @dense_reward: If True, then return dense reward in range 0 to 10, only applicable for query_for_preference=False.
            If False, then return sparse reward (i.e., 0 or 1) based in 'threshold_score'.
        @threshold_score: Used to decide sparse reward.
    """

    if dense_reward:
        assert query_for_preference is False, f"Dense reward only supports for single sequence query."

    custom_rewards = np.zeros_like(actions)
    executed_actions = actions 
    MAX_ATTEMPT = 3  # Recommended from: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/api-errors#handle-errors
    
    timesteps_to_check=len(actions)

    timesteps_check = []
    
    shift_steps = 1  # Hyper-parameter
    gemini_texts = []
    total_times_query = 0
    total_times_query_success = 0


    if timesteps_to_check> 0:
        for timestep in range(timesteps_to_check):
            ATTEMPTED_TO_AVOID_BLOCK = False
            #if object_actions[timestep] != 0:  # Check if 'object action' is None
            # We set the window size equal to 'default_window_size' for all types of low-level tasks
            window_size = timestep if timestep < default_window_size else default_window_size
            
            
            # ================= Construct the image contains sequence of observations to query VLM =================
            if include_start_state:
                if window_size==default_window_size:
                    seq_1 = raw_observations[timestep - window_size+1 : timestep +2]   
                else:
                    seq_1 = raw_observations[timestep - window_size : timestep +2] 
                
                if query_for_preference:
                    seq_2 = raw_observations[timestep - window_size + 1 - shift_steps: timestep + 2 - shift_steps]  
                    image_to_ask_vlm = create_image_from_2_sequences(seq_1, seq_2, window_size + 1)
                else: 
                    if window_size < default_window_size:
                        image_to_ask_vlm =  create_single_image_without_wall(seq_1, edit_image=False, obs_size=256)
                    else:
                        image_to_ask_vlm =  create_single_image_without_wall(seq_1, edit_image=False, obs_size=256)

            else:
                seq_1 = raw_observations[timestep - window_size + 2: timestep + 2] 
                if query_for_preference:
                    seq_2 = raw_observations[timestep - window_size + 2 - shift_steps: timestep + 2 - shift_steps] 
                    image_to_ask_vlm = create_image_from_2_sequences(seq_1, seq_2, window_size)
                else:
                    image_to_ask_vlm =  create_single_image_without_wall(seq_1, edit_image=False, obs_size=256)

            # ================= Construct the string contains sequence of actions to query VLM =================
            if window_size < default_window_size:
                action_seq_1 = [CUSTOMIZED_ACTION_VOCAB[executed_actions[i]] for i in range(timestep - window_size, timestep+1)]
            else:
                action_seq_1 = [CUSTOMIZED_ACTION_VOCAB[executed_actions[i]] for i in range(timestep - window_size, timestep+1)][-5:]

            if query_for_preference:
                action_seq_2 = [CUSTOMIZED_ACTION_VOCAB[executed_actions[i]] for i in range(timestep - window_size, timestep)]
                actions_for_prompt = {'seq_1': action_seq_1, 'seq_2': action_seq_2}
            else:
                actions_for_prompt = {'seq_1': action_seq_1}
            query_success = False
            attempt_cnt = 0
            while not query_success:
                gemini_flash, gemini_pro, gemini_key = construct_vlm_with_key(rank=rank)
                probability_distribution, gemini_text, error = query_probability_from_vlm_with_images(
                    #vlm_model=vlm_model,
                    gemini_flash=gemini_flash, 
                    gemini_pro=gemini_pro,
                    image=image_to_ask_vlm,
                    actions=actions_for_prompt,
                    low_level_instruction=instruction,
                    window_size=window_size,
                    include_start_state=include_start_state,
                    query_for_preference=query_for_preference,
                    dense_reward=dense_reward,
                    threshold_score=threshold_score,
                    sleep=sleep,
                    data_idx=data_idx,
                    timestep=timestep
                )
                total_times_query += 1
                attempt_cnt += 1

                if error is None:
                    query_success = True
                    total_times_query_success += 1
                    gemini_texts.append(gemini_text)
                else:
                    if attempt_cnt == MAX_ATTEMPT:
                        print(f"[INFO] Ignore querying Gemini after tried {MAX_ATTEMPT}.")
                        break

                    if isinstance(error, BlockedPromptException):
                        if ATTEMPTED_TO_AVOID_BLOCK:
                            print(f"[INFO] Got 'Block' again, ignore after attempt to fix (e.g., image edit).")
                            break

                        print(f"[INFO] Got 'block_reason: OTHER' from Gemini, attempt to edit image to avoid. (rank={rank}, {gemini_key})")
                        print(f"===> Error from Gemini: {error}, LLI: {instruction}")
                        if query_for_preference:
                            image_to_ask_vlm = create_image_from_2_sequences(seq_1, seq_2, window_size, edit_image=True)
                        else:
                            image_to_ask_vlm = create_image_from_1_sequence(seq_1, window_size, edit_image=True)
                        ATTEMPTED_TO_AVOID_BLOCK = True

                    elif error.grpc_status_code.name in ["DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL", "UNKNOWN", "UNAVAILABLE"]:
                        # Reference error: https://cloud.google.com/apis/design/errors#handling_errors
                        print(f"[INFO] Got '{error.grpc_status_code.name}' from Gemini, retrying ({attempt_cnt}) ... (rank={rank}, {gemini_key})")
                        time.sleep(4)

                    else:
                        print(f"[INFO] Unknown error {error}, re-trying... (rank={rank}, {gemini_key})")
                        time.sleep(4)

            timesteps_check.append(timestep)

    query_info = dict(
        timesteps_check=timesteps_check,
        total_times_query=total_times_query,
        total_times_query_success=total_times_query_success,
        gemini_texts=gemini_texts,
    )
    return query_info

def get_vlm_reward_wrapper(params):
    obs_size = 256
    raw_obs, action, instruction, at_last_step, include_start_state, query_for_preference, rank, sleep, data_idx = params

    query_info = get_reward_with_vlm(
        raw_obs, action, instruction,
        at_last_step=at_last_step,
        include_start_state=include_start_state,
        default_window_size=0,
        query_for_preference=query_for_preference,
        dense_reward=False,
        threshold_score=10,
        rank=rank,
        sleep=sleep,
        obs_size=obs_size,
        data_idx=data_idx
    )
    timesteps_checked, gemini_texts = query_info["timesteps_check"], query_info["gemini_texts"]
    total_times_query_success = query_info["total_times_query_success"]

    return timesteps_checked, total_times_query_success, gemini_texts, data_idx

