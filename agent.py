import asyncio
import json
import os

try:
    from pydantic.v1 import BaseModel, Field
except:
    from pydantic import BaseModel, Field

from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

from langchain.agents import AgentExecutor, ConversationalChatAgent, Tool
from langchain.agents.agent import ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain.chains import LLMMathChain
from langchain_community.callbacks import get_openai_callback
# from langchain.chat_models.base import BaseChatModel
# from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
# from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
#                                          CallbackManagerForChainRun)
from langchain_core.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                              CallbackManagerForChainRun)
from langchain_core.exceptions import OutputParserException
# from langchain.tools.base import BaseTool
from langchain_core.tools import BaseTool
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from image_search import ImageQuery, ImageSearch
from KGQnA import DaiGraphCypherQAChain, DaiNeo4jGraph
# from langchain.memory import ConversationBufferWindowMemory
from memory import (ConversationUserBufferWindowMemory,
                    RedisUserChatMessageHistory)

"""
Define the tools that will be used by the agent
"""

# KGQnA_desc = """Utilize this tool to get a FINAL TABULAR answer to simple fact-based queries made by the user that are specifically related to vendors, items, customers, sales representatives, sales regions and store records.
# All queries to this tool must start with one of the following phrases: what, which, how many, who, how much, list e.g. `List the top 10 items sold by perdue farms.`, `which vendor sells troyer chicken breast?`, `how many cases were bought by the winey cow in 2021?`, `how many order cases were sold in orders made to willy's salsa?`, `what is the item info of items sold by vendor smithfield?`.
# This is NOT a search tool. It is tool that returns the answer directly to the user. All queries to this tool must be about a single subject.
# DO NOT use when you need to perform some action on the answer or if the answer has to be human-readable text.
# Input should be full question (preferably, the user's original question). Do NOT include any agent instructions or any other information not explicitly mentioned."""

# ftsearch_desc = """Utilize this tool as a search tool to get context for queries that are about vendors, items, or customers. This specialised tool offers search capabilities to get you information about any relevant entities.
# Use this tool for cases where you need to formulate a human-readable response with some reasoning or perform some comparison.
# Input should be a comma-seperated list of entities with their type: <entity_type> <entity> e.g. query: `how should I sell cheesecake cone to martins?`, input: `item cheesecake cone , customer martins`.
# Do not include any agent instructions or any other information not explicitly mentioned."""

KGQnA_desc = """This tool is a QnA tool that returns a tabular answer for some fact-based queries specifically related to vendors, \
items, customers, sales representatives, sales regions and store records from data stored in the user's Knowledge Graph.
It's output is returned directly to user i.e. return_direct=True so Assistant cannot use the output from this tool.
Utilize this tool ONLY to get a FINAL TABULAR answer to fact-based queries made by the user. DO NOT use this tool for comparisons or to get an opinion.
Some example queries to this tool are: `List the top 10 items sold by perdue farms.`, `which vendor sells troyer chicken breast?`, \
`how many cases were bought by the winey cow in 2021?`, `how many order cases were sold in orders made to willy's salsa?`
DO NOT use when you need to perform some action on the answer or if the answer has to be human-readable text.
Input should be the user's original question. Do NOT include any agent instructions or any other information not explicitly mentioned."""

KGSearch_desc = """This tool is a search tool that gets context from user's Knowledge Graph for answering queries that are about vendors, items, or customers.
This specialised tool offers search capabilities and returns a list of dictionaries as response that can be used by Assistant as context for answering user's queries.
Input should be a single fact-based search query string. Use this tool when you need to do some reasoning or calculations on the search result e.g. find the difference or sum."""

imgsearch_desc = """This tools is a search tool that returns a list of item images/flyers relevant to the user's query \
when asked to provide images or flyers in the query. Only use this tool when user explicitly asks for images in their query.
An example query to this tool is: `Give me all the images about Atsuri`. Input should be the user's original question."""

imgquery_desc = """This tool is a Q/A tool that is used to answer a query about an image/flyer. Only use this tool when user explicitly \
mentions an image or flyer e.g. by mentioning '...in this flyer' or '...in this image' in their query.
Examples: `Give me a brief marketing report based on the contents of this item image.`, `List all the details of the vendor mentioned in this flyer.`, `Give me a sales pitch that I can use along with this item image.`, `What is the manufacturer name mentioned in this image?` """

class CalculatorInput(BaseModel):
    question: str = Field()

class KGQnAInput(BaseModel):
    query: str = Field(description="A single string containing only the question")


class KGSearchInput(BaseModel):
    query: str = Field(description="A one-line string containing the search query")

class ImgSearchInput(BaseModel):
    query: str = Field(description="A single one-line string containing the user's query")

# In particular, Assistant specialises in answering questions on vendors, customers, stores, items, sales representatives, sales regions, sales records and purchase order records. \
# Assistant is good at giving accurate advice when asked using all information available.

SYSTEM_MESSAGE = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. \
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. \
Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Assistant is designed to think step-by-step and to break down the user's queries into smaller steps to solve before responding.

Assistant is designed to provide detailed responses that leave no room for doubt or confusion.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."""

HUMAN_MESSAGE = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. \
Assistant should remember that tools with return_direct=True answer directly to the user and cannot be used combined with other tools. \
The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}
"""

class QnA_Agent(AgentExecutor):
    save2memory: bool = True
    mem_token_limit: int = 1000
    save_direct: bool = True
    filters_run: Dict[str, str] = {}
    image_run: str = ''
    memory: Optional[ConversationUserBufferWindowMemory] = None
    
    @staticmethod
    def function_name():
        return "KGQnA_Agent"

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
        answer = outputs.get(self.output_keys[0], "")
        if isinstance(answer, dict):
            outputs = {
                "heading": answer.get("heading", ""),
                "rows": answer.get("rows", ""),
                "suggestions": answer.get("suggestions", []),
                "images_urls": answer.get("images_urls", []),
                "llm_response": answer.get("llm_response", ""),
            }
        elif isinstance(answer, str):
            if answer == "":
                outputs = {
                    "heading": "",
                    "rows": "",
                    "suggestions": [],
                    "llm_reponse": "",
                    "images_urls": []
                }
            else:
                try:
                    answer = json.loads(answer)
                    outputs = {
                        "heading": answer.get("heading", ""),
                        "rows": answer.get("rows", ""),
                        "suggestions": answer.get("suggestions", []),
                        "images_urls": answer.get("images_urls", []),
                        "llm_response": "",
                    }
                except:
                    # This means that it is probably not a json string
                    outputs = {
                        "heading": "",
                        "rows": "",
                        "suggestions": [],
                        "llm_response": answer,
                        "images_urls": []
                    }
        else:
            # print(type(answer))
            outputs = {
                "heading": "Internal Server Error!",
                "rows": "",
                "suggestions": [],
                "llm_reponse": "",
                "images_urls": []
            }

        if (
            self.memory is not None
        ):  # Only save to memory if this agent has a memory object
            if self.save2memory:  # Only save to memory if it has to be saved
                # self.mem_token_limit sets a limit on the number of tokens to save at each call. Prevents exceeding model token limit
                if outputs["llm_response"] != "":
                    doutput = {
                        self.output_keys[0]: outputs["llm_response"][:self.mem_token_limit]
                    }
                elif outputs["rows"] != "":
                    doutput = f"{outputs['heading']}\n{outputs['rows'][:self.mem_token_limit]}"
                    doutput = {self.output_keys[0]: doutput}
                elif len(outputs['images_urls']) > 0:
                    images_urls = '\n'.join(outputs['images_urls'])
                    doutput = f"{outputs['heading']}\n{images_urls}"
                    doutput = {self.output_keys[0]: doutput}
                else:
                    doutput = {self.output_keys[0]: ""}
                self.memory.save_context(
                    inputs,
                    doutput
                )
            else:
                self.memory.save_context(inputs, {self.output_keys[0]: ""})

        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

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
            # This `else` Block Is My Own Changes Made To Allow For Any Of 3 Possible Keynames
            #######################################################################
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                _input_keys = _input_keys.difference(self.memory.memory_variables)

            _input_keys = list(_input_keys)
            if len(_input_keys) == 0:
                raise ValueError("No agent inputs found in input!")
            
            user_id = inputs.pop('user_id', '-1')
            

            query = inputs.pop("question", None)
            if query is None:
                query = inputs.pop("query", None)
                if query is None:
                    query = inputs.get(_input_keys[0], None)
                    if query is None:
                        raise ValueError(
                            f"""The inputs dictionary should have one of either of these 3 keys: \
                            `question`, `query` or `{_input_keys[0]}`"""
                        )
            filters = inputs.pop('filters', None)
            self.filters_run = None
            if filters is not None and len(filters) > 0:
                self.filters_run = filters
            
            self.image_run = None
            image = inputs.pop('image_path', None)
            if image is not None and len(image) > 0:
                self.image_run = image
            inputs = {_input_keys[0]: query, 'user_id': user_id}
            #######################################################################
            # END OF CHANGES
            #######################################################################

        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs

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
                self.save2memory = True
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                    ####################################################################################
                    # THIS IS MY PERSONAL MODIFICATION: IN CASE OF return_direct: return its output directly as AgentFinish
                    ####################################################################################
                    self.save2memory = self.save_direct
                    # print(self.save2memory)
                    # output = tool.run(
                    # agent_action.tool_input,
                    # verbose=self.verbose,
                    # color=color,
                    # callbacks=run_manager.get_child() if run_manager else None,
                    # **tool_run_kwargs,
                    # )
                    # print(output,type(output))
                    # return AgentFinish({'output':output},"")
                    #####################################################################################
                    # END OF CHANGES
                    #####################################################################################
                # We then call the tool on the tool input to get an observation
                
                tool_input = agent_action.tool_input
                if "KG" in tool.name or "KG" in agent_action.tool:
                    # if not isinstance(tool_input, dict):
                    #     tool_input =  f"QUESTION: {tool_input}"
                    if isinstance(tool_input, dict):
                        tool_input = f"{tool_input.values[0]}"
                    if self.filters_run is not None and len(self.filters_run) > 0:
                        # tool_input['filters'] = self.filters_run
                        tool_input += f" FILTERS: {json.dumps(self.filters_run)}"
                elif tool.name == 'Image Q/A' or agent_action.tool == 'Image Q/A':
                    if isinstance(tool_input, dict):
                        tool_input = f"{tool_input.values[0]}"
                    if self.image_run is not None and len(self.image_run) > 0:
                        tool_input += f" IMAGE: {self.image_run}"
                observation = tool.run(
                    tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
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
                    ####################################################################################
                    # THIS IS MY PERSONAL MODIFICATION: IN CASE OF RETURN_DIRECT, return its output as AgentFinish
                    ####################################################################################
                    self.save2memory = self.save_direct
                    # output = await tool.arun(
                    # agent_action.tool_input,
                    # verbose=self.verbose,
                    # color=color,
                    # callbacks=run_manager.get_child() if run_manager else None,
                    # **tool_run_kwargs,
                    # )
                    # return AgentFinish({'output':output},"")
                    #####################################################################################
                    # END OF CHANGES
                    #####################################################################################

                # We then call the tool on the tool input to get an observation
                tool_input = agent_action.tool_input
                if "KG" in tool.name or "KG" in agent_action.tool:
                    # if not isinstance(tool_input, dict):
                    #     tool_input =  f"QUESTION: {tool_input}"
                    if isinstance(tool_input, dict):
                        tool_input = f"{tool_input.values[0]}"
                    if self.filters_run is not None and len(self.filters_run) > 0:
                        # tool_input['filters'] = self.filters_run
                        tool_input += f" FILTERS: {json.dumps(self.filters_run)}"
                elif tool.name == 'Image Q/A' or agent_action.tool == 'Image Q/A':
                    if isinstance(tool_input, dict):
                        tool_input = f"{tool_input.values[0]}"
                    if self.image_run is not None and len(self.image_run) > 0:
                        tool_input += f" IMAGE: {self.image_run}"
                observation = await tool.arun(
                    tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await InvalidTool().arun(
                    agent_action.tool,
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
    def initialize(cls, image_search: ImageSearch, image_query: ImageQuery, verbose: bool = False, **kwargs):
        
        llm1 = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", max_tokens=500)
        llm2 = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', max_tokens=1000)
        # memory = ConversationTokenBufferMemory(memory_key="chat_history",llm=llm,max_token_limit=500, return_messages=True)
        # memory = ConversationBufferWindowMemory(
        #     memory_key="chat_history", k=5, return_messages=True
        # )
        
        memory = ConversationUserBufferWindowMemory(
            chat_memory=RedisUserChatMessageHistory("s1", key_prefix="kgqna_message_store"),
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        neo_host = os.environ.get('GL_HOST',"dai-mlops-neo4j-service")
        neo_port = int(os.environ.get('GL_PORT',7687))
        
        if neo_host is None:
            raise Exception("GL_HOST environment variable not found! Can't continue")
        
        neo_passwd = os.environ.get('GL_PASSWORD',None)
        neo_user = os.environ.get('GL_USER','neo4j')
        
        math_tool = LLMMathChain.from_llm(llm2)

        graph = DaiNeo4jGraph(
            url=f"bolt://{neo_host}:{neo_port}", username=neo_user, password=neo_passwd
        )

        kg_qna = DaiGraphCypherQAChain.from_llm(
            graph=graph, llm=llm1, qa_llm=llm2, verbose=verbose
        )

        tools = [
            Tool.from_function(
                func=math_tool.run,
                coroutine=math_tool.arun,
                name="Calculator",
                description="useful for when you need to answer questions about math or arithmetic",
                args_schema=CalculatorInput
            ),
            Tool.from_function(
                func=kg_qna.answer_question,
                coroutine=kg_qna.async_answer_question,
                args_schema=KGQnAInput,
                name="KG QnA",
                description=KGQnA_desc,
                return_direct=True,
            ),
            Tool.from_function(
                func=kg_qna.search,
                coroutine=kg_qna.asearch,
                args_schema=KGSearchInput,
                name="KG Search",
                description=KGSearch_desc,
            ),
            Tool.from_function(
                func=image_search.search,
                coroutine=image_search.asearch,
                args_schema=ImgSearchInput,
                name="Image Search",
                description=imgsearch_desc,
                return_direct=True
            ),
            Tool.from_function(
                func=image_query.query,
                coroutine=image_query.aquery,
                args_schema=ImgSearchInput,
                name="Image Q/A",
                description=imgquery_desc,
                return_direct=True
            )
        ]

        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm1,
            tools=tools,
            system_message=SYSTEM_MESSAGE,
            human_messge=HUMAN_MESSAGE,
            output_parser=kwargs.get("output_parser", None),
        )

        agent_chain = cls.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=verbose,
            memory=memory,
            max_iterations=kwargs.get("max_iterations", 5),
            max_execution_time=kwargs.get(
                "max_execution_time", 30
            ),  # wait for a max 5s
            handle_parsisng_errors=kwargs.get(
                "handle_parsing_errors",
                True,
            ),
            return_intermediate_steps=False,
        )

        # agent_chain = initialize_agent(tools,
        # llm,
        # agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        ## agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # memory=memory,
        # verbose=verbose,
        # agent_kwargs={'human_message':HUMAN_MESSAGE,'system_message':SYSTEM_MESSAGE},
        # max_iterations=kwargs.get("max_iterations",2),
        # max_execution_time=kwargs.get("max_execution_time",10), # wait for a max 5s
        # handle_parsisng_errors=kwargs.get("handle_parsing_errors",True)
        # )

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.save2memory = True
        return super().run(*args, **kwargs)


def answer_question_agent(query: str, agent, print_stats=False):
    with get_openai_callback() as cb:
        try:
            answer = agent.run(query)
        except Exception as e:
            # print(type(e))
            return {
                "heading": f"Sorry, there was some problem answering your query: '{e}'",
                "rows": "",
                "suggestions": [],
                "llm_response": f"Sorry, there was some problem answering your query: '{e}'",
            }
        if print_stats:
            print("--" * 50)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print("--" * 50)

    try:
        answer = json.loads(answer)
        return {
            "heading": answer.get("heading", ""),
            "rows": answer.get("rows", ""),
            "suggestions": answer.get("suggestions", []),
            "llm_response": "",
        }
    except:
        # This means that it is probably not a json string
        return {"heading": "", "rows": "", "suggestions": "", "llm_response": answer}
