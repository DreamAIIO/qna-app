from __future__ import annotations

import json
import os
# import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

# from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key
# from langchain.utilities.redis import get_client
from langchain_community.utilities.redis import get_client
# from langchain.schema import BaseChatMessageHistory
# from langchain.schema.messages import (AIMessage, BaseMessage, HumanMessage,
#                                        _message_to_dict, get_buffer_string,
#                                        messages_from_dict)
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     get_buffer_string, message_to_dict,
                                     messages_from_dict)

try:
    from pydantic.v1 import Field
except:
    from pydantic import Field

# logger = logging.getLogger(__name__)


class BaseUserChatMessageHistory(ABC):
    """Abstract base class for storing chat message history for each user (separately).

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A list of Messages stored in-memory or in a database or filesystem."""

    def add_user_message(self, message: str, user_id: str, k: int) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message), user_id=user_id, k=k)

    def add_ai_message(self, message: str, user_id: str, k: int) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        self.add_message(AIMessage(content=message), user_id=user_id, k=k)

    @abstractmethod
    def add_message(self, message: BaseMessage, user_id: str, k: int) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self, **kwargs) -> None:
        """Remove all messages from the store"""


class RedisUserChatMessageHistory(BaseUserChatMessageHistory):
    """Chat message history stored in a Redis database."""

    def __init__(
        self,
        session_id: str,
        url: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        url = url or os.environ.get("REDIS_URL","localhost")
        port = port or os.environ.get("REDIS_PORT",'6379')
        url = f"redis://{url}:{port}/0"
        try:
            self.redis_client = get_client(redis_url=url)
        except redis.exceptions.ConnectionError as error:
            # logger.error(error)
            print(error)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    # @property
    def messages(self, k: int, user_id: str) -> List[BaseMessage]:  # type: ignore
        """Retrieve the last 'k' messages of the user from Redis"""
        _items = self.redis_client.lrange(self.key+user_id, 0, k-1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        # print(items)
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage, user_id: str, k: int) -> None:
        """Append the message to the user's record in Redis"""
        # print(message, user_id)
        self.redis_client.lpush(self.key+user_id, json.dumps(message_to_dict(message)))
        self.redis_client.ltrim(self.key+user_id, 0, k)
        if self.ttl:
            self.redis_client.expire(self.key+user_id, self.ttl)

    def clear(self, user_id) -> None:
        """Clear session memory of user from Redis"""
        for k in self.redis_client.scan_iter(f"{self.key}{user_id}*"):
            self.redis_client.delete(k)

class ConversationUserBufferWindowMemory(BaseChatMemory):
    """Buffer for storing conversation memory inside a limited size window."""

    chat_memory: BaseUserChatMessageHistory = Field(default_factory=RedisUserChatMessageHistory)
    # output_key: Optional[str] = None
    # input_key: Optional[str] = None
    # return_messages: bool = False

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    user_id_key: str = "user_id" #: :meta private:
    k: int = 5
    """Number of messages to store in buffer."""
    '''
    def __init__(
        self, 
        chat_memory: Optional[BaseUserChatMessageHistory] = None,
        k: int = 5, 
        memory_key: str = 'history', 
        user_id_key: str = 'user_id',
        human_prefix: str = "Human", 
        ai_prefix: str = "AI",
        return_messages: bool = False,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ):
        if not chat_memory:
            self.chat_memory = RedisUserChatMessageHistory(session_id='s1')
        else:
            self.chat_memory = chat_memory
        self.input_key = input_key
        self.output_key = output_key
        self.k = k
        self.memory_key = memory_key
        self.user_id_key = user_id_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.return_messages = return_messages
    '''

    def _get_input_output(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    
    def buffer(self, user_id) -> Dict[str, Union[str, List[BaseMessage]]]:
        return self.buffer_as_messages(user_id) if self.return_messages else self.buffer_as_str(user_id)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        user_id = inputs.pop(self.user_id_key, "")
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str, user_id, k=self.k*2)
        self.chat_memory.add_ai_message(output_str, user_id, k=self.k*2)

    # @property
    def buffer_as_str(self, user_id: str) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        messages = self.chat_memory.messages(self.k*2, user_id)
        # messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    # @property
    def buffer_as_messages(self, user_id) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        # return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return self.chat_memory.messages(self.k*2, user_id)

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        user_id = inputs.get(self.user_id_key, '')
        return {self.memory_key: self.buffer(user_id)}
        # return {self.memory_key: self.buffer}
    
    def clear(self, key):
        self.chat_memory.clear(key)
