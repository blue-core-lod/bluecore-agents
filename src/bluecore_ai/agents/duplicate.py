import os
from dataclasses import dataclass
from typing import Union

import httpx
import rdflib

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pymilvus import MilvusClient

from bluecore_ai.agents.base import get_token
from bluecore_models.utils.graph import BF, init_graph, load_jsonld

BLUECORE_API_URL = os.getenv("BLUECORE_API_URL", "http://localhost:3000")


@dataclass
class SupportDependencies:
    milvus_client: MilvusClient
    embedding_func: callable
    incoming_rdf: str


class DeDupResult(BaseModel):
    score: float  # Should be a percentage
    best_match: str


agent = Agent(
    instructions=(
        "You are an expert on deciding if a BIBFRAME Work or Instance "
        "already exists in Blue Core. You will take incoming Work or Instance RDF URI and call the "
        "{retrieve} tool that returns the Blue Core embeddings that best match "
        "the incoming RDF triples. From these embeddings determine a score that "
        "aggregates the best matched triples and then return a score and the "
        " URL of the Blue Core Work or Instance."
    ),
    output_type=DeDupResult,
)

INSTANCE_SPARQL = """SELECT DISTINCT ?s WHERE { ?s a bf:Instance . FILTER(isIRI(?s))}"""
WORKS_SPARQL = """SELECT DISTINCT ?s WHERE { ?s a bf:Work . FILTER(isIRI(?s)) }"""


def get_subject_url_and_type(graph: rdflib.Graph) -> tuple:
    work_query = graph.query(WORKS_SPARQL, initNs={"bf": BF})
    works = [work[0] for work in work_query]
    if len(works) == 1:
        return (str(works[0]), "works")
    instance_query = graph.query(INSTANCE_SPARQL, initNs={"bf": BF})
    instances = [instance[0] for instance in instance_query]
    if len(instances) == 1:
        return (str(instances[0]), "instances")


@agent.tool
async def retrieve(context: RunContext[SupportDependencies]):
    incoming_rdf = context.deps.incoming_rdf
    if incoming_rdf.startswith("http"):
        entity_result = httpx.get(incoming_rdf)
        entity = entity_result.json()
        incoming_graph = load_jsonld(entity["data"])
        entity_uri = entity["uri"]
        bf_type = entity["type"]
    else:  # Try to parse string as json_ld
        incoming_graph = init_graph()
        incoming_graph.parse(data=incoming_rdf, format="json-ld")
        entity_uri, bf_type = get_subject_url_and_type(incoming_graph)

    skolemized_graph = incoming_graph.skolemize(basepath=f"{entity_uri}#")
    triples = [
        line.strip(".") for line in skolemized_graph.serialize(format="nt").splitlines()
    ]

    triple_vectors = context.deps.embedding_func.encode_documents(triples)
    search_result = context.deps.milvus_client.search(
        collection_name=bf_type, data=triple_vectors, output_fields=["uri", "text"]
    )
    output = {}
    for result in search_result:
        for row in result:
            uri = row["entity"]["uri"]
            if uri in output:
                output[uri].append(
                    {"distance": row["distance"], "triple": row["entity"]["text"]}
                )
            else:
                output[uri] = [
                    {"distance": row["distance"], "triple": row["entity"]["text"]}
                ]
    return output
