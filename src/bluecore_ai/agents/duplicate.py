from dataclasses import dataclass
from typing import Union

from pydantic_ai import Agent, RunContext


@dataclass
class SupportDependencies:
    model: str
    bluecore_api_url: str


agent = Agent(
    result_type=bool,
    system_prompt=(
        'You are an expert on deciding if a BIBFRAME Work or Instance '
        'already exists in our datastore using any of the available Blue '
        'Core API {endpoints} using the {is_duplicate} tool.'
    )
)

@agent.tool
async def endpoints(
    ctx: RunContext[SupportDependencies]
) -> dict:
    return {}

@agent.tool
async def is_duplicate(
    ctx: RunContext[SupportDependencies]
) -> Union[bool, float]:
    return True
    
