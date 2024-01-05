"""The SovaAI Conversation integration."""
from __future__ import annotations

import logging
from typing import Literal
import json
import yaml
import aiohttp


from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL, ATTR_NAME
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.util import ulid
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
    ServiceNotFound,
)

from homeassistant.helpers import (
    config_validation as cv,
    intent,
    template,
    entity_registry as er,
)

from .const import (
    CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DOMAIN,
)

from .exceptions import (
    EntityNotFound,
    EntityNotExposed,
    CallServiceError,
    FunctionNotFound,
    NativeNotFound,
    FunctionLoadFailed,
    ParseArgumentsFailed,
    InvalidFunction,
)

from .helpers import (
    FUNCTION_EXECUTORS,
    FunctionExecutor,
    NativeFunctionExecutor,
    ScriptFunctionExecutor,
    TemplateFunctionExecutor,
    RestFunctionExecutor,
    ScrapeFunctionExecutor,
    CompositeFunctionExecutor,
    convert_to_template,
    get_function_executor,
)


_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# hass.data key for agent.
DATA_AGENT = "agent"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up SovaAI Conversation from a config entry."""

    agent = SovaAIAgent(hass, entry)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload SovaAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class SovaAIAgent(conversation.AbstractConversationAgent):
    """SovaAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        base_url = entry.data.get(CONF_BASE_URL)
        self.client = aiohttp.ClientSession(base_url)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        exposed_entities = self.get_exposed_entities()
        context = { "env_sitelang": "ru-RU", "isDevice": True }

        if user_input.conversation_id in self.history:
            async with self.client.post('/api/Chat.init',json={"uuid": "6944d0b0-ca59-4007-97bf-867d6c4385a9", "cuid": user_input.conversation_id, "context": context}) as response:
               json_response = await response.json()
               conversation_id = json_response['result']['cuid']
        else:
            async with self.client.post('/api/Chat.init',json={"uuid": "6944d0b0-ca59-4007-97bf-867d6c4385a9", "cuid": "", "context": context}) as response:
               json_response = await response.json()
               conversation_id = json_response['result']['cuid']
        user_input.conversation_id = conversation_id
        try:
            response = await self.query(user_input, context, exposed_entities, 0)
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response['content'])
        return conversation.ConversationResult(
            response=intent_response, conversation_id=response['conversation_id']
        )

    def _async_generate_prompt(self, raw_prompt: str, exposed_entities) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
            },
            parse_result=False,
        )

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def get_functions(self):
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def query(
        self,
        user_input: conversation.ConversationInput,
        context,
        exposed_entities,
        n_requests,
    ):
        """Process a sentence."""
#        functions = list(map(lambda s: s["spec"], self.get_functions()))
#        function_call = "auto"
#        if len(functions) == 0:
#            functions = None
#            function_call = None

        _LOGGER.info("Prompt: %s", user_input.text)

        async with self.client.post('/api/Chat.request',json={"cuid": user_input.conversation_id, "text": user_input.text, "context": context}) as response:
            _LOGGER.info("Response %s", response)
            json_response = await response.json()
            user_input.conversation_id = json_response['result']['cuid']
            message = {"content": json_response['result']['text']['value'], "conversation_id": json_response['result']['cuid'], "context": json_response['result']['context']}
            return message


#        choice: Choice = response.choices[0]
#        message = choice.message
#        if choice.finish_reason == "function_call":
#            message = await self.execute_function_call(
#                user_input, messages, message, exposed_entities, n_requests + 1
#            )

    def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ):
        function_name = message.function_call.name
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ):
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(message.function_call.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": message.function_call.name,
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)
