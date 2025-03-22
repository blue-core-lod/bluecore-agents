"""Module for a Validation Generative AI Agent for Blue Core"""

import pathlib


from dataclass import dataclass
from typing import Union

import rdflib

from pydantic_ai import Agent, RunContext
from pyshacl import validate


@dataclass
class SupportDependencies:
    model: str
    bluecore_api_url: str
    shacl: Union[rdflib.Graph, None]


agent = Agent(
    result_type=bool,
    system_prompt=(
        'You are an expert on deciding if a BIBFRAME Work or Instance '
        'is valid according to the SHACL graphs and by current resources '
        'in the Blue Core vector space. A critical test is the {validate} '
        'tool using any of the available Blue Core API {endpoints}.'
    )
)

@agent.tool
async def endpoints(
    ctx: RunContext[SupportDependencies]
) -> dict:
    return {}

@agent.tool
async def validate(
    ctx: RunContext[SupportDependencies]
) -> Union[bool, rdflib.Graph]:
    return True
    


