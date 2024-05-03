import asyncio
# import json
import inspect
import os
import re
# from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tenacity import (AsyncRetrying, RetryError, Retrying, retry_if_result,
                      stop_after_attempt)

try:
    from pydantic.v1 import BaseModel, Field
except:
    from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent import ExceptionTool
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.tools import InvalidTool
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.schema import RUN_KEY
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.manager import (AsyncCallbackManager,
                                              AsyncCallbackManagerForChainRun,
                                              CallbackManager,
                                              CallbackManagerForChainRun)
from langchain_core.exceptions import OutputParserException
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables import (RunnableConfig, RunnableSerializable,
                                      ensure_config, run_in_executor)
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
# from langchain_community.callbacks import get_openai_callback
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from memory import (ConversationUserBufferWindowMemory,
                    RedisUserChatMessageHistory)
from PDF_Tools import (PDFQnA, PDFQnAFile,  # DEFAULT_COLLECTION_NAME
                       UserFiles, get_pdfs_index, update_data)
from vector_stores import create_collection


class CalculatorInput(BaseModel):
    question: str = Field()

class QnAInput(BaseModel):
    query: str = Field(description="A string containing the input query")

PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Assistant is very polite and interactive and always keeps the user engaged. When Assistant does not know the answer, Assistant responds as such and asks the user for more information.

When a tool returns an apology or an error message, Assistant should also respond with an apology to the user and ask the user for more information depending on the tool's message.

Assistant always keeps track of the past conversation and uses it to answer any questions to the best of its abilities. When user asks for a list or a tool responds with a list, Assistant always responds with a numbered or a bullet list in its final answer.

Assistant makes good use of whitespaces and its response is both easy and enjoyable to read. Assistant is always detailed and elaborate in its reponses.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist. Assistant always finishes its response with 'Thanks for asking!'."""


SUFFIX = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. Assistant should ask the user to use other tools to lookup information if the first tool does not help. Assistant does not need to inform the user if a tool failed or which tool it used to get to an answer. The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""

pdfqna_desc = """This tool is a QnA tool that you can use to answer questions of all kinds from an existing knowledge base. \
Use this tool if there is a 'filtering expression' e.g. "which...", "that...", in the your question or if it is a question about a single 'subject' or a single 'page' of the file. Also use this tool if the question is about the number of pages, words etc.
Input should be a SINGLE STRING containing the user question.
EXAMPLE INPUTS:
1. `What are the selling points for Perdue Chicken given on page 13?`.
2. `What are all the details of ribeye pork chops from the smithfield case ready manual?`
3. `List all the details of item ribeye pork chops.`
Do NOT include any agent instructions or anything other than the user query in the input."""

pdfqna_desc1 = """This tool is a QnA and summarising tool that you can use to answer long questions which use ALL the content from a particular file given a filename.
Use this tool to answer questions with no 'filtering expression' or queries asking for a summary of the whole file.
Do NOT use this tool for answering single fact-based queries or queries about a particular subject.
Input should be a SINGLE STRING containing the question.
EXAMPLE INPUTS: 
1. `List all items mentioned in the smithfield case ready manual.`
2. `Summarise the custom lipari.pdf file.`
3. `Extract all the salient points from Smithfield Case Ready Manual.`
4. `List all the details of all the items mentioned in the smithfield case ready manual.`
Do NOT include any agent instructions or anything other than the user query in the input."""


class QnA_Agent(AgentExecutor):
    
    vector_store: VectorStore
    # pdf_qna: PDFQnAFile
    run_source: str = ""
    filenames_run: List[str] = []
    user_id_run: str = ""
    user_files: UserFiles
    rephraser: Optional[LLMChain] = None
    memory: Optional[ConversationUserBufferWindowMemory] = None
    # user_files_loaded: Dict[str, set]

    @staticmethod
    def function_name():
        return "QnA_Agent"
    
#     @property
#     def input_keys(self) -> List[str]:
#         """Return the input keys.

#         :meta private:
#         """
#         return self.agent.input_keys + ['user_id']
    
    
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                    
                # We then call the tool on the tool input to get an observation
                ################################################################################
                # THIS IS MY PERSONAL MODIFICATION: If the tool is PDF QnA Tool, also pass the #
                # filenames and user_id as input to the tool that this agent recieved as input #
                ################################################################################
                tool_input = agent_action.tool_input
                if "PDF" in tool.name or "PDF" in agent_action.tool:
                    if not isinstance(tool_input, dict):
                        tool_input = f"QUESTION: {tool_input}"
                    else:
                        tool_input = f"QUESTION: {tool_input.values[0]}"
                    
                    if self.filenames_run is not None and len(self.filenames_run) > 0:
                        if 'FILENAMES' in tool_input:
                            tool_input += ',' + ','.join(self.filenames_run)
                        else:
                            tool_input += f" FILENAMES: {','.join(self.filenames_run)}"
                    if self.user_id_run is not None and len(self.user_id_run) > 0:
                        tool_input += f" USER ID: {self.user_id_run}"    

                observation = tool.run(
                    tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                ####################################################################################
                # THIS IS MY PERSONAL MODIFICATION: If the tool is PDF QnA tool, separate sources
                ####################################################################################
                if "PDF" in tool.name or "PDF" in agent_action.tool: # If this is the PDF Q/A tool, we need to store source
                    #print(f"Observation: {observation}")
                    answer = re.split(r"SOURCES?:",observation, maxsplit=1)
                    observation = answer[0].strip()
                    source = ""
                    if len(answer) > 1:
                        source = answer[1].split('\n')[0].strip()
                    self.run_source = source
                    #observation = answer
                    
                    
                    #print(f"ANSWER: {observation}\nSOURCE: {self.run_source}")
                #####################################################################################
                # END OF CHANGES
                #####################################################################################
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        
        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                ################################################################################
                # THIS IS MY PERSONAL MODIFICATION: If the tool is PDF QnA Tool, also pass the #
                # filenames and user_id as input to the tool that this agent recieved as input #
                ################################################################################
                tool_input = agent_action.tool_input
                if "PDF" in tool.name or "PDF" in agent_action.tool:
                    if not isinstance(tool_input, dict):
                        tool_input = f"QUESTION: {tool_input}"
                    else:
                        tool_input = f"QUESTION: {tool_input.values[0]}"
                    
                    if self.filenames_run is not None and len(self.filenames_run) > 0:
                        if 'FILENAMES' in tool_input:
                            tool_input += ',' + ','.join(self.filenames_run)
                        else:
                            tool_input += f" FILENAMES: {','.join(self.filenames_run)}"
                    if self.user_id_run is not None and len(self.user_id_run) > 0:
                        tool_input += f" USER ID: {self.user_id_run}"
                    #print(tool_input)
                observation = await tool.arun(
                    tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                ####################################################################################
                # THIS IS MY PERSONAL MODIFICATION: If the tool is PDF QnA tool, separate sources
                ####################################################################################
                if "PDF" in tool.name or "PDF" in agent_action.tool: # If this is the PDF Q/A tool, we need to store source
                    answer = re.split(r"SOURCES?:",observation, maxsplit=1)
                    observation = answer[0].strip()
                    source = ""
                    if len(answer) > 1:
                        source = answer[1].split('\n')[0].strip()
                    self.run_source = source
                    #observation = answer
                    
                #####################################################################################
                # END OF CHANGES
                #####################################################################################
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            return agent_action, observation

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)
    
    @classmethod
    def initialize(cls, verbose: bool = False, rephraser: Optional[LLMChain] = None, **kwargs):

        llm = ChatOpenAI(temperature=0,model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-4-0125-preview"),max_tokens=2500) 
        # llm2 = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0,max_tokens=1000)
        #'gpt-3.5-turbo'
        #memory = ConversationTokenBufferMemory(memory_key="chat_history",llm=llm,max_token_limit=500, return_messages=True)
        # memory = ConversationBufferWindowMemory(memory_key="chat_history",k=10,return_messages=True)
        memory = None
        if kwargs.get("save_to_memory", True):
            memory = ConversationUserBufferWindowMemory(
                chat_memory=RedisUserChatMessageHistory("s1", key_prefix="pdfqna_message_store"),
                k=8,
                memory_key="chat_history",
                return_messages=True
            )
        
        index_name = kwargs.get("index_type","milvus")
        if index_name.lower() == 'milvus':
            col_created = create_collection(drop_old=kwargs.get("drop_old_collection", False))
            if col_created:
                print("New collection created!")
        vector_index = get_pdfs_index(index_name=index_name) #collection_name=DEFAULT_COLLECTION_NAME
        print("Vector Index Loaded!")
        search_kwargs = {'k': 8, 'fetch_k': 30, 'lambda_mult': 0.3}
        if index_name == "faiss":
            search_kwargs['fetch_k'] = 20
            # search_kwargs["score_threshold"] = 0.2
        tables_folder = os.environ.get("TABLES_FOLDERPATH", "table_files")
        if not os.path.isdir(tables_folder):
            os.mkdir(tables_folder)

        user_files = UserFiles(vector_index)

        pdf_qna = PDFQnA.from_llm_and_vector_index(llm=llm, user_files=user_files, vector_index=vector_index, chain_type="stuff", search_kwargs=search_kwargs, verbose=verbose)
        
        pdf_qna_file = PDFQnAFile.from_llm_and_vector_index(llm=llm, user_files=user_files, vector_index=vector_index, verbose=verbose, token_max=3500)
        
        math_tool = LLMMathChain.from_llm(llm)

        tools = [
            Tool.from_function(
                func=math_tool.run,
                coroutine=math_tool.arun,
                name="Calculator",
                description="useful for when you need to answer questions about math or arithmetic",
                args_schema=CalculatorInput
            ),
            Tool.from_function(
                func=pdf_qna.run,
                coroutine=pdf_qna.arun,
                name="PDF Q/A",
                return_direct=False,
                description=pdfqna_desc,
                args_schema=QnAInput
            ),
            Tool.from_function(
                func=pdf_qna_file.run,
                coroutine=pdf_qna_file.arun,
                name="PDF Whole File Q/A & Summariser",
                return_direct=True,
                description=pdfqna_desc1,
            ),
        ]
        
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            input_variables=['input', 'chat_history', 'agent_scratchpad'],
            system_message=PREFIX,
            human_message=SUFFIX,
            output_parser=kwargs.get('output_parser',None)
        )
        
        agent_chain = cls.from_agent_and_tools(agent=agent,
                                               tools=tools, 
                                               verbose=verbose,
                                               memory=memory,
                                               #max_iterations=kwargs.get("max_iterations",10),
                                               #max_execution_time=kwargs.get("max_execution_time",60), # wait for a max 60s
                                               handle_parsisng_errors=kwargs.get("handle_parsing_errors","Check your output and make sure it conforms!"),
                                               return_intermediate_steps=False,
                                               early_stopping_method="generate",
                                               vector_store=vector_index,
                                               user_files=user_files,
                                               rephraser=rephraser
                                            #    user_files_loaded=user_files,
                                               #pdf_qna=pdf_qna_file
                                            )
        
        return agent_chain
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def input_keys(self):
        return ['query', 'user_id', 'filenames']
    
    @property
    def output_keys(self):
        return ['output']

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prepare chain inputs, including adding inputs from memory.

        Args:
            inputs: Dictionary of raw inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the chain's
                memory.

        Returns:
            A dictionary of all inputs, including those added by the chain's memory.
        """
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        else:
            #######################################################################
            # This `else` Block Is My Own Changes Made to Incorporate user_id and filenamess
            #######################################################################
            missing_keys = set(self.input_keys).difference(inputs)
            if missing_keys:
                # print(inputs.keys())
                raise ValueError(f"Missing some input keys: {missing_keys}")
            
            # _input_keys = []
            # for ik in self.input_keys:
            #     if memory is None or not ik in self.memory.memory_variables:
            #         _input_keys.append(ik)
            
            # if len(_input_keys) == 0:
            #     raise ValueError("No agent inputs found in input!")
            
            query = inputs.pop('query',None) or inputs['input']
            user_id = inputs.get('user_id', "-1")
            filenames = inputs.pop('filenames',[])
            # filenames = ','.join(filenames)
            if user_id is not None:
                self.user_id_run = str(user_id).split('_')[0]
                # query = f"QUESTION: {query} USER ID: {user_id}"
            else:
                self.user_id_run = "u1"
                # query = f"QUESTION: {query}"
            #print(self.user_id_run)
            if filenames is not None and len(filenames) > 0:
                self.filenames_run = filenames
                # query = f"{query} FILENAMES: {filenames}"
            else:
                self.filenames_run = ""

            # inputs[_input_keys[0]] = query
            inputs = {
                'input': query,
                'user_id': user_id
            }
            #######################################################################
            # END OF CHANGES
            #######################################################################
        
        self.run_source = "" # My own personal addition: Set run_source to empty string before the chain is called.
        
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        
        # self._validate_inputs(inputs)
        return inputs
    
    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prepare chain outputs, and save info about this run to memory.

        Args:
            inputs: Dictionary of chain inputs, including any inputs added by chain
                memory.
            outputs: Dictionary of initial chain outputs.
            return_only_outputs: Whether to only return the chain outputs. If False,
                inputs are also added to the final outputs.

        Returns:
            A dict of the final chain outputs.
        """
        self._validate_outputs(outputs)
        # My own personal addition: If run_source has been set, then add it to answer
        if self.run_source != "":
            ans_key = list(outputs.keys())[0]
            outputs[ans_key] = f"{outputs[ans_key]}\n\nSOURCES: {self.run_source}"
            self.run_source = ""
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
    
    def rephrase(self, question):
        requestion = self.rephraser.invoke(question, return_only_outputs=True)[self.rephraser.output_keys[0]]
        requestion_ext = re.findall(r"Output: (.+)", requestion, flags=re.I)
        if len(requestion_ext) == 0:
            return requestion
        return requestion_ext[0]
    
    async def async_rephrase(self, question):
        requestion = await self.rephraser.ainvoke(question, return_only_outputs=True)[self.rephraser.output_keys[0]]
        requestion_ext = re.findall(r"Output: (.+)", requestion, flags=re.I)
        if len(requestion_ext) == 0:
            return requestion
        return requestion_ext[0]
    
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        def retry_check_fn(a):
            retry_terms = ['sorry', "don't know", "do not know", "does not contain", "doen't contain",
                          "can't find", "cannot find", "unable to find", "does not discuss", "no documents found"]
            return any([rt in a.lower() for rt in retry_terms])
        
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name") or self.get_name()
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        inputs = self.prep_inputs(input)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            name=run_name,
        )
        try:
            for attempt in Retrying(stop=stop_after_attempt(2), retry=retry_if_result(retry_check_fn)):
                with attempt:
                    outputs = (
                        self._call(inputs, run_manager=run_manager)
                        if new_arg_supported
                        else self._call(inputs)
                    )
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(outputs[self.output_keys[0]].lower())
                    if self.rephraser is not None and retry_check_fn(outputs[self.output_keys[0]].lower()):
                        q = inputs['input']
                        inputs['input'] = self.rephrase(q)
                        # print(q, inputs['input'])
        except RetryError:
            pass
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs
    
    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        def retry_check_fn(a):
            retry_terms = ['sorry', "don't know", "do not know", "does not contain", "doesn't contain",
                          "can't find", "cannot find", "unable to", "does not discuss", "no documents found"]
            return any([rt in a.lower() for rt in retry_terms])
        
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name") or self.get_name()
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        inputs = self.prep_inputs(input)
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        output_key = self.output_keys[0] or 'output'
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            name=run_name,
        )
        
        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(2), retry=retry_if_result(retry_check_fn)):
                with attempt:
                    try:
                        outputs = (
                            await self._acall(inputs, run_manager=run_manager)
                            if new_arg_supported
                            else await self._acall(inputs)
                        )
                    except BaseException as e:
                        print(e)
                        await run_manager.on_chain_error(e)
                        outputs = {output_key: f"Sorry, there was a problem answering this query! Try asking me something else. I'm happy to assist."}
                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(outputs[output_key].lower())
                    if self.rephraser is not None and retry_check_fn(outputs[output_key].lower()):
                        q = inputs['input']
                        inputs['input'] = self.rephrase(q)
        except RetryError:
            pass
        await run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs

    def update_data(self, file_ids: List[str], filenames: Optional[List[str]]=None, user_id:Optional[Union[int,str]]=None,
                    filepaths :Optional[List[str]]=None, data: Optional[Union[bytes, str]]=None,
                    chunk_size:int=2000, use_ocr:bool=False, use_docai:bool=True) -> None:
        user_id = str(user_id)
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        if filenames is None and filepaths is not None:
            filenames = [filepath.split('/')[-1] for filepath in filepaths]
        if isinstance(filenames, str):
            filenames = [filenames]
        status, message = update_data(self.vector_store, filenames=filenames, file_ids=file_ids, user_id=user_id, filepaths=filepaths, data=data, chunk_size=chunk_size, use_ocr=use_ocr, use_docai=use_docai)
        if status:
            for filename, file_id in zip(filenames, file_ids):
                self.user_files.remove_file(file_id, user_id)
                self.user_files.add_file(file_id, filename, user_id=user_id)
        return status, message
    
    def delete_files(self, file_ids: List[str], user_id: str):
        store_type = str(type(self.vector_store)).lower()
        if "milvus" in store_type.lower():
            ids_to_delete = []
            sfile_ids = [f"'{fn.lower()}'" for fn in file_ids]
            sfile_ids = '[' + ','.join(sfile_ids) + ']'
            ids_to_delete = self.vector_store.query(
                expr=f"id in {sfile_ids} and user_id == '{user_id}'",
                return_ids=True,
                return_data=False,
                k=None
            )
            if len(ids_to_delete) > 0: # Delete all the ids to be deleted after new docs have been added
                _ = self.vector_store.delete(
                    ids=ids_to_delete
                )
                print("File docs deleted!")
                
            #print(ids)
            # print(ids_to_delete)
            
        elif "chroma" in store_type.lower():
            try:
                sfile_ids = [f.lower() for f in file_ids]
                self.vector_store.delete(
                    filter={'id': {"$in": sfile_ids}, 'user_id': user_id}
                )
                print("File docs deleted!")
                
            except BaseException as be:
                False, f"There was a problem in deleting the docs for files: {file_ids} due to:\n{be}"
        else:
            raise NotImplementedError("Only milvus and chroma vector stores support file deletion so far.")
        for file_id in file_ids:
            self.user_files.remove_file(file_id, user_id)
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    VERBOSE = True

    pdfqna_agent = QnA_Agent.initialize(verbose=VERBOSE)
    print("PDFQnA Agent Created!")
    con = True
    user_id = "u1_1"
    while con:
        choice = input("Please select one of two options:\n[1: Answer query\n2: Upload file\n3: Quit]\nEnter the option number: ")
        if choice == '1':
            filenames = input("Please enter a comma-separated list of filenames you want to query: ")
            filenames = filenames.strip().split(',')
            query = input("Please enter your query: ")
            pdfqna_agent.invoke({"query": query, "filenames": filenames, "user_id": user_id})
        elif choice == '2':
            filenames = input("Please enter a comma-separated list of filepaths (GCP URI) you want to query: ")
            filenames = filenames.split(',')
        elif choice == '3':
            con = False
        else:
            print("Invalid Option Selected\n")
        