"""Module for a Validation Generative AI Agent for Blue Core"
import dataclass
import pathlib

import rdflib
@dataclass
class SHACL:
    graph: rdflib.Graph
