import json

import pytest
import rdflib

from unittest.mock import Mock

from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pymilvus import model, MilvusClient
from pytest_httpx import HTTPXMock

from bluecore_models.utils.vector_db import init_collections

from bluecore_ai.agents.duplicate import (
    retrieve,
    SupportDependencies,
    agent as dedeup_agent,
)
from bluecore_models.utils.graph import init_graph, BF

models.ALLOW_MODEL_REQUESTS = False


WORK_URL = "https://bcld.info/works/34d72304-3d21-4860-b67f-3a32dc42e057"


def _setup_vectordb(work_graph, tmp_path):
    db_path = tmp_path / "vector.db"
    milvus_client = MilvusClient(str(db_path))
    embedding_func = model.DefaultEmbeddingFunction()

    init_collections(milvus_client)

    skolemized_graph = work_graph.skolemize(basepath=f"{WORK_URL}#")

    triples = [
        line.rstrip(".")
        for line in skolemized_graph.serialize(format="nt").splitlines()
    ]

    triple_vectors = embedding_func.encode_documents(triples)

    embeddings_data = []

    for i, vector in enumerate(triple_vectors):
        embeddings_data.append(
            {
                "id": i,
                "vector": vector,
                "text": triples[i],
                "uri": WORK_URL,
                "version": 1,
            }
        )

    milvus_client.insert(collection_name="works", data=embeddings_data)

    return milvus_client, embedding_func


@pytest.fixture
def work_graph():
    graph = init_graph()
    test_work = rdflib.URIRef(WORK_URL)
    graph.add((test_work, rdflib.RDF.type, BF.Work))
    return graph


@pytest.mark.asyncio
async def test_deduplicate_agent_zero_score(work_graph):
    # Create mock dependencies
    mock_milvus_client = Mock()
    mock_embedding_func = Mock()

    # Set up mock return values
    mock_embedding_func.encode_documents.return_value = [
        [0.1] * 128
    ]  # Mock embedding vector
    mock_milvus_client.search.return_value = [[]]  # Mock empty search results

    deps = SupportDependencies(
        milvus_client=mock_milvus_client,
        embedding_func=mock_embedding_func,
        incoming_rdf=work_graph.serialize(format="json-ld"),
    )

    with capture_run_messages() as messages:
        dedeup_agent.model = TestModel(
            custom_output_args={"score": 0.0, "best_match": ""}
        )
        result = await dedeup_agent.run("Process the incoming RDF data", deps=deps)

    assert messages
    assert result.output.score == 0.0


@pytest.mark.asyncio
async def test_retrieve_string(work_graph, tmp_path):
    milvus_client, embedding_func = _setup_vectordb(work_graph, tmp_path)

    deps = SupportDependencies(
        milvus_client=milvus_client,
        embedding_func=embedding_func,
        incoming_rdf=work_graph.serialize(format="json-ld"),
    )
    context = Mock()
    context.deps = deps

    result = await retrieve(context)

    assert result[WORK_URL][0]["distance"] == 1.0
    assert result[WORK_URL][0]["triple"].startswith(
        f"<{WORK_URL}> <{rdflib.RDF.type}> <{BF.Work}>"
    )


@pytest.mark.asyncio
async def test_retrieve_url(work_graph, tmp_path, httpx_mock: HTTPXMock):
    loc_work_url = "http://id.loc.gov/resources/works/23293669"
    loc_work_uri = rdflib.URIRef(loc_work_url)
    loc_work_graph = init_graph()
    loc_work_graph.add((loc_work_uri, rdflib.RDF.type, BF.Work))

    httpx_mock.add_response(
        method="GET",
        url=loc_work_url,
        json={
            "uri": loc_work_uri,
            "type": "works",
            "data": json.loads(loc_work_graph.serialize(format="json-ld")),
        },
    )

    milvus_client, embedding_func = _setup_vectordb(work_graph, tmp_path)

    deps = SupportDependencies(
        milvus_client=milvus_client,
        embedding_func=embedding_func,
        incoming_rdf=loc_work_url,
    )

    context = Mock()
    context.deps = deps

    result = await retrieve(context)

    assert result[WORK_URL][0]["distance"] == pytest.approx(0.86, rel=1e-1)
