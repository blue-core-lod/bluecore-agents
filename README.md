# Blue Core Generative AI Agents
Repository of Generative AI Agents used by the [Blue Core API](https://github.com/blue-core-lod/bluecore_api) 
and [Blue Core Workflows](https://github.com/blue-core-lod/bluecore-workflows) built with
[Pydantic AI](https://ai.pydantic.dev/). 


## Duplicate Agent
To use the de-duplicate Agent in a Jupyter notebook:

1. Import supporting standard packages
1. Import the Pydantic AI and Pymilvus modules
1. Import the agent, SupportDependencies, and bluecore models helper functions
1. Instantiate a LLM model using environmental variables and set as agent's model

```python
import asyncio
import os

import rdflib

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pymilvus import model, MilvusClient

from bluecore_ai.agents.duplicate import SupportDependencies, agent as dedup_agent
from bluecore_models.utils.graph import init_graph, BF

llm_model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        base_url=os.environ.get("LLM_URL"),  # For an OpenAI compatible API provider, omit if using OpenAI's API
        api_key=os.environ.get("OPENAI_API_KEY")  # Can be omitted if using OpenAI API and set in local env
    )
)
dedup_agent.model = llm_model
nest_asyncio.apply()  # Allows you to use asyncio calls in a notebook
```

1. Load a RDF graph or use an URL that is compatible with the Blue Core API
1. Set `SupportDependencies` with pymilvus client and embedding function, and either the serialized RDF as json-ld or
   RDF URL
1. Run the agent

```python
graph = init_graph()
graph.load(data="test-entity.rdf")  # For a local RDF XML file

deps = SupportDependencies(
    milvus_client=MilvusClient(url="http://localhost:19530"),  # Running the Blue Core stack locally
    embedding_func=model.DefaultEmbeddingFunction(),
    incoming_rdf=graph.serialize(format='json-ld')
)

result = dedup_agent.run_sync("Please evaluate the incoming RDF", deps=deps)
```
The result will have a `DeDupResult` in the output property with a score and the best matched URL in the Blue 
Core Datastore:

```python
result.output

> DeDupResult(score=0.9463598132133484, best_match='https://dev.bcld.info/works/82dbf27a-0a28-4707-bfe9-e49a17286a21')
```
 
## Validation Agent 


