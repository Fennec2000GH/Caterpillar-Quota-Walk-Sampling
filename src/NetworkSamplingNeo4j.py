
from neo4j import GraphDatabase

uri = 'bolt://localhost:1080'
driver = GraphDatabase.driver(uri=uri, auth=('user', 'password'), param_config=None)

