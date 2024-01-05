"""Config flow for SovaAI Conversation integration."""
from __future__ import annotations

import logging
import types
import yaml
from types import MappingProxyType
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME, CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
    AttributeSelector,
    SelectSelector,
    SelectSelectorConfig,
    SelectOptionDict,
    SelectSelectorMode,
)

from .const import (
    CONF_FUNCTIONS,
    CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONF_BASE_URL,
    DOMAIN,
    DEFAULT_NAME,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
    }
)

DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    base_url = data.get(CONF_BASE_URL)

    if base_url == DEFAULT_CONF_BASE_URL:
        # Do not set base_url if using SovaAI for case of SovaAI's base_url change
        base_url = None
        data.pop(CONF_BASE_URL)



class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for SovaAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """SovaAI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )
        schema = self.sovaai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )

    def sovaai_config_option_schema(self, options: MappingProxyType[str, Any]) -> dict:
        """Return a schema for SovaAI completion options."""
        if not options:
            options = DEFAULT_OPTIONS

        return {
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS)},
                default=DEFAULT_CONF_FUNCTIONS_STR,
            ): TemplateSelector(),
        }
