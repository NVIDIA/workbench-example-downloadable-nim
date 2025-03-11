# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import re
import os
import time
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(layout="wide")

# Define constants
provider_list = ["meta",
                 "mistralai",
                 "nv-mistralai",
                 "nvidia",
                 "google",
                 "microsoft",
                 "institute-of-science-tokyo", 
                 "tokyotech-llm", 
                 "qwen",
                 "yentinglin", 
                 "deepseek-ai"]

meta_model_list = ["llama-3.1-8b-instruct",
                   "llama-3.1-70b-instruct",
                   "llama-3.1-405b-instruct",
                   "llama-3.3-70b-instruct",
                   "llama3-8b-instruct",
                   "llama3-70b-instruct",
                   "codellama-70b"]

mistralai_model_list = ["mistral-7b-instruct-v0.3", 
                        "mixtral-8x7b-instruct-v0.1",
                        "mixtral-8x22b-instruct-v0.1"]

nv_mistralai_model_list = ["mistral-nemo-12b-instruct"]

nvidia_model_list = ["llama-3.1-nemotron-70b-instruct"]

google_model_list = ["gemma-2-9b-it"]

microsoft_model_list = ["phi-3-mini-4k-instruct"]

institute_of_science_tokyo_model_list = ["llama-3.1-swallow-8b-instruct-v0.1",
                                         "llama-3.1-swallow-70b-instruct-v0.1"]

tokyotech_llm_model_list = ["llama-3-swallow-70b-instruct-v0.1"]

qwen_model_list = ["qwen2.5-7b-instruct"]

yentinglin_model_list = ["llama-3-taiwan-70b-instruct"]

deepseek_ai_model_list = ["deepseek-r1"]

sample_queries = ["Write a limerick about the wonders of GPU computing.",
                  "What can I see at NVIDIA's GPU Technology Conference?",
                  "Tell me about Dumbledore."]

discard_prefix = ["mistral-nemo-12b-instruct"] # Some downloadable NIMs discard the provider in the model name

# Default recommendation for other GPUs
default_recommendation = {
    "1": {
        "models": ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"],
    }, 
    "2": {
        "models": ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"],
    }, 
    "4": {
        "models": ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"],
    }, 
    "8": {
        "models": ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"],
    }, 
    "16": {
        "models": ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"],
    }, 
}

system_prompt = """
You are a helpful AI assistant for question-answering tasks. If you don't know the answer, just say that you don't know. Be as helpful as you can while answering the question thoroughly, maintaining a good balance between conciseness and verbosity in your response. 

If the question is a greeting, always answer it without using the context.
"""

# Global Helper Functions

def num_gpu_helper(num):
    if num == 1:
        return "1"
    if num < 4 and num >= 2:
        return "2"
    elif num < 8:
        return "4"
    elif num < 16:
        return "8"
    else:
        return "16"

def rtx_filter(model, gpu_type):
    rtx_list = ["RTX 5090", "RTX 5080", "RTX 4090", "RTX 4080"]
    return model + "-RTX" if gpu_type in rtx_list else model

def get_compose_instructions(provider, model, gpu_type):
    model = rtx_filter(model, gpu_type)
    return f"""
    1. Open the AI Workbench Project Dashboard. 
    2. Select the ``{provider}/{model}`` model from the Compose profile dropdown
    3. Start Compose

    **Note:** It may take several minutes to download the NIM container and model weights. 
    
    You can monitor compose progress by clicking **Outputs** on the bottom left corner and selecting "Compose" from the dropdown. """

def get_troubleshooting_instructions():
    return """
    * If you haven't started the NIM, select **Compose** > ``[model-profile]`` > **Start**. Monitor logs under **Output** on the bottom left of the AI Workbench window. 
    * If hitting a 401 error, you may need to run a ``docker login nvcr.io`` on the host to authenticate to the NGC registry. 
    * If hitting a cache permissions error, you may need to run a ``sudo chmod -R a+w ~/.cache/nim`` on the host to elevate permissions."""

def create_chat_model(provider, llm, use_local, gpu_type):
    if use_local:
        if llm == "llama-3.1-swallow-8b-instruct-v0.1" or llm == "llama-3.1-swallow-70b-instruct-v0.1":
            provider = "tokyotech-llm" # not institute-of-science-tokyo if local
        if llm == "codellama-70b":
            llm = "codellama-70b-instruct" # need instruct if local
        llm = rtx_filter(llm, gpu_type)
        return ChatOpenAI(
            model_name=provider + "/" + llm if llm not in discard_prefix else llm,
            openai_api_key=os.environ["NVIDIA_API_KEY"],
            openai_api_base=f"http://downloadable-nim-{llm}-1:8000/v1"
        )
    if os.getenv("NVIDIA_INTERNAL") == "True":
        nvdev_endpoint = "nvdev/" + provider + "/" + llm
        return ChatNVIDIA(model=nvdev_endpoint)
    else:
        endpoint = provider + "/" + llm
        return ChatNVIDIA(model=endpoint)
    
def create_chat_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()

def string_to_stream(text, delay=0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

def reset_models():
    if st.session_state.selected_provider == "meta":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = meta_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "llama-3.1-8b-instruct"
    elif st.session_state.selected_provider == "mistralai":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = mistralai_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "mistral-7b-instruct-v0.3"
    elif st.session_state.selected_provider == "nv-mistralai":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = nv_mistralai_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "mistral-nemo-12b-instruct"
    elif st.session_state.selected_provider == "nvidia":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = nvidia_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "llama-3.1-nemotron-70b-instruct"
    elif st.session_state.selected_provider == "google":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = google_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "gemma-2-9b-it"
    elif st.session_state.selected_provider == "microsoft":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = microsoft_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "phi-3-mini-4k-instruct"
    elif st.session_state.selected_provider == "institute-of-science-tokyo":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = institute_of_science_tokyo_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "llama-3.1-swallow-8b-instruct-v0.1"
    elif st.session_state.selected_provider == "tokyotech-llm":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = tokyotech_llm_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "llama-3-swallow-8b-instruct-v0.1"
    elif st.session_state.selected_provider == "qwen":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = qwen_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "qwen2.5-7b-instruct"
    elif st.session_state.selected_provider == "yentinglin":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = yentinglin_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "llama-3-taiwan-70b-instruct"
    elif st.session_state.selected_provider == "deepseek-ai":
        st.session_state.selected_provider = st.session_state.selected_provider
        st.session_state.model_list = deepseek_ai_model_list
        st.session_state.selected_model = st.session_state.selected_model if st.session_state.selected_model in st.session_state.model_list else "deepseek-r1"

def reset_gpus():
    st.session_state.gpu_type = st.session_state.gpu_type
    st.session_state.gpu_num = st.session_state.gpu_num
    reset_models()

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gpu_type" not in st.session_state:
    st.session_state.gpu_type = ""
if "gpu_num" not in st.session_state:
    st.session_state.gpu_num = 1
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "disk_space" not in st.session_state:
    st.session_state.disk_space = "Estimated disk space required is currently unavailable for this model."
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = "meta"
if "model_list" not in st.session_state:
    st.session_state.model_list = meta_model_list
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instruct"
if "selected_query" not in st.session_state:
    st.session_state.selected_query = None
if "use_local" not in st.session_state:
    st.session_state.use_local = False

# Title and Header
st.title("Chat with NVIDIA NIM")
    
# Add this after the title and before the info message
col_gpu1, col_gpu2 = st.columns([1, 2], gap="medium")
with col_gpu1:
    deployment_type = st.radio(
        "How would you like to run the models?",
        ["Use Remote Endpoints", "Use a NIM on the Host GPU"],
        index=None,
        key="deployment_type",
        on_change=reset_models
    )

    if deployment_type == "Use a NIM on the Host GPU":
        st.session_state.use_local = True
        col_local1, col_local2 = st.columns([2, 1])
        with col_local1:
            st.selectbox(
                "What type of GPU do you have? ([more](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html))",
                [
                    "H100",
                    "A100 80GB",
                    "A100 40GB",
                    "L40S",
                    "A10G",
                    "RTX 6000 Ada",
                    "RTX 5090",
                    "RTX 5080",
                    "RTX 4090",
                    "RTX 4080",
                    "Other NVIDIA GPU",
                    "Not sure"
                ],
                index=None,
                key="gpu_type",
                placeholder="Type your GPU model...",
                on_change=reset_gpus
            )

        with col_local2:
            st.number_input(
                "How many?",
                min_value=1,
                max_value=16,
                value="min",
                key="gpu_num",
                step=1,
                on_change=reset_gpus
            )
        
        # Add GPU recommendations section
        if st.session_state.gpu_type and st.session_state.gpu_num:
            
            def update_recommendation():
                # Define recommendations based on GPU type
                with open('support_matrix.json', 'r') as file:
                    gpu_recommendations = json.load(file)
                    gpu_type = st.session_state.gpu_type
                    gpu_num = st.session_state.gpu_num
                    size = gpu_recommendations.get(gpu_type, default_recommendation)
                    recommendation = size.get(num_gpu_helper(gpu_num))
                    
                    # update recommendation
                    if st.session_state.selected_model in recommendation["models"] and gpu_type != "Other NVIDIA GPU" and gpu_type != "Not sure":
                        st.session_state.recommendation = "👍 Nice! The current model is expected to be compatible with your selected hardware. "
                    elif st.session_state.selected_model not in recommendation["models"] and gpu_type != "Other NVIDIA GPU" and gpu_type != "Not sure":
                        st.session_state.recommendation = "🛑 Warning! The current model may not be compatible with your selected hardware. Consider selecting a different model. "
                    elif gpu_type == "Other NVIDIA GPU" or gpu_type == "Not sure":
                        st.session_state.recommendation = "⚠️ We recommend starting with smaller models for optimal local performance. Monitor your GPU memory in AI Workbench usage when running locally. "
    
                # Define estimated disk size based on model and GPU
                with open('disk_size.json', 'r') as file:
                    disk_size = json.load(file)
                    
                    # update disk space
                    try:
                        throughput = disk_size[st.session_state.selected_model]["disk_space"][gpu_type]["fp16"]["throughput"]
                        st.session_state.disk_space = f"This model takes approximately {throughput} GB of disk space"
                    except:
                        st.session_state.disk_space = "Estimated disk space required is currently unavailable for this model."
            
            st.markdown("---")
            update_recommendation()
            
            if "Nice" in st.session_state.recommendation:
                st.success(st.session_state.recommendation)
                st.markdown("---")
                with st.expander("Start the Downloadable NIM"):
                    st.write(get_compose_instructions(st.session_state.selected_provider, 
                                                      st.session_state.selected_model,
                                                      st.session_state.gpu_type))
                with st.expander("Troubleshooting"):
                    st.write(get_troubleshooting_instructions())
            elif "Warning" in st.session_state.recommendation:
                st.error(st.session_state.recommendation)
            else:
                st.warning(st.session_state.recommendation)
                st.markdown("---")
                with st.expander("Start the Downloadable NIM"):
                    st.write(get_compose_instructions(st.session_state.selected_provider, 
                                                      st.session_state.selected_model,
                                                      st.session_state.gpu_type))
                with st.expander("Troubleshooting"):
                    st.write(get_troubleshooting_instructions())
    
    elif deployment_type == "Use Remote Endpoints":
        st.session_state.use_local = False
        st.markdown("---")
        st.info("You'll be using NVIDIA AI Foundation Models through our cloud endpoints. This gives you access to all available models without local GPU requirements.")

with col_gpu2:
    # Replace the current if deployment_type condition with this:
    if deployment_type:
        if deployment_type == "Use Remote Endpoints" or (deployment_type == "Use a NIM on the Host GPU" and st.session_state.gpu_type):
            # Container for message history
            with st.container():
                message_history = st.container(height=300)
    
                # Render Chat History
                for message in st.session_state.messages:
                    with message_history.chat_message(message["role"]):
                        st.markdown(message["content"])
    
                # Chatbot Settings
                col1, col2 = st.columns([2,2])
                with col1:
                    if "selected_provider" not in st.session_state:
                        st.selectbox('Select your model provider:', 
                                     provider_list, 
                                     index=0,
                                     key="selected_provider", 
                                     on_change=reset_models)
                    else:
                        st.selectbox('Select your model provider:', 
                                     provider_list, 
                                     index=provider_list.index(st.session_state.selected_provider),
                                     key="selected_provider", 
                                     on_change=reset_models)
    
                with col2:
                    def update_recommendation():
                        st.session_state.selected_model = st.session_state.selected_model
                        
                        # Define recommendations based on GPU type
                        with open('support_matrix.json', 'r') as file:
                            gpu_recommendations = json.load(file)
                            gpu_type = st.session_state.gpu_type
                            gpu_num = st.session_state.gpu_num
                            size = gpu_recommendations.get(gpu_type, default_recommendation)
                            recommendation = size.get(num_gpu_helper(gpu_num))
                            
                            # update recommendation
                            if st.session_state.selected_model in recommendation["models"] and gpu_type != "Other NVIDIA GPU" and gpu_type != "Not sure":
                                st.session_state.recommendation = "👍 Nice! The current model is expected to be compatible with your selected hardware. "
                            elif st.session_state.selected_model not in recommendation["models"] and gpu_type != "Other NVIDIA GPU" and gpu_type != "Not sure":
                                st.session_state.recommendation = "🛑 Warning! The current model may not be compatible with your selected hardware. Consider selecting a different model. "
                            elif gpu_type == "Other NVIDIA GPU" or gpu_type == "Not sure":
                                st.session_state.recommendation = "⚠️ We recommend starting with smaller models for optimal local performance. Monitor your GPU memory in AI Workbench usage when running locally. "
            
                        # Define estimated disk size based on model and GPU
                        with open('disk_size.json', 'r') as file:
                            disk_size = json.load(file)
                            
                            # update disk space
                            try:
                                throughput = disk_size[st.session_state.selected_model]["disk_space"][gpu_type]["fp16"]["throughput"]
                                st.session_state.disk_space = f"This model takes approximately {throughput} GB of disk space"
                            except:
                                st.session_state.disk_space = "Estimated disk space required is currently unavailable for this model."

                    if "selected_model" not in st.session_state:
                        st.selectbox('Choose a model:', 
                                     st.session_state.model_list, 
                                     index=0,
                                     key="selected_model",
                                     help=st.session_state.disk_space if (deployment_type == "Use a NIM on the Host GPU") else None,
                                     on_change=update_recommendation)
                    else:
                        st.selectbox('Choose a model:', 
                                     st.session_state.model_list, 
                                     index=st.session_state.model_list.index(st.session_state.selected_model),
                                     key="selected_model",
                                     help=st.session_state.disk_space if (deployment_type == "Use a NIM on the Host GPU") else None,
                                     on_change=update_recommendation)
    
                query_selector = st.pills("Say something like", sample_queries, on_change=reset_models)
    
                # Define model and chains
                llm = create_chat_model(st.session_state.selected_provider, 
                                       st.session_state.selected_model, 
                                       st.session_state.use_local,
                                       st.session_state.gpu_type)
                chain = create_chat_chain(llm)
    
                if query_selector != st.session_state.selected_query:
                    if query_selector is None:
                        pass
                    else: 
                        # A sample query was selected, processing
                        with message_history.chat_message("user"):
                            st.markdown(query_selector)
        
                        with message_history.chat_message("assistant"):
                            try: 
                                response = st.write_stream(chain.stream({"input": query_selector}))
                            except Exception as e:
                                error_msg = "Something went wrong: " + str(e)
                                if str(e) == "Connection error." and st.session_state.use_local:
                                    error_msg += " Are you sure this NIM is running on your host GPU?"
                                response = st.write_stream(string_to_stream(error_msg))
                                st.markdown("**Troubleshooting Tips**")
                                st.markdown(get_troubleshooting_instructions())
                    
                        st.session_state.messages.append({"role": "user", "content": query_selector})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                    st.session_state.selected_query = query_selector
                
                # Generate Response
                if prompt := st.chat_input("What would you like to chat about?"):
                    if prompt is None:
                        pass
                    else: 
                        with message_history.chat_message("user"):
                            st.markdown(prompt)
                        
                        with message_history.chat_message("assistant"):
                            try: 
                                response = st.write_stream(chain.stream({"input": prompt}))
                            except Exception as e:
                                error_msg = "Something went wrong: " + str(e)
                                if str(e) == "Connection error." and st.session_state.use_local:
                                    error_msg += " Are you sure this NIM is running on your host GPU?"
                                response = st.write_stream(string_to_stream(error_msg))
                                st.markdown("**Troubleshooting Tips**")
                                st.markdown(get_troubleshooting_instructions())
                    
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.write("")
            st.info("Please select your GPU type to begin chatting.")
    else:
        st.write("")
        st.info("Please select how you would like to run the models to begin chatting.")
