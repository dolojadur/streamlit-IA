"""
Health-check para Neo4j.
Uso:
  python check_neo4j.py
Lee variables desde ST secrets (si existe) o desde variables de entorno:
  NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
Devuelve 0 si OK o 1 si hay error.
"""
import os
import sys
try:
    from neo4j import GraphDatabase
except Exception as e:
    print("ERROR: falta el paquete 'neo4j'. Instalalo con: pip install neo4j")
    sys.exit(1)

NEO4J_URL = os.environ.get("NEO4J_URL", "neo4j+s://c63dbf3f.databases.neo4j.io")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

print(f"Comprobando conectividad a: {NEO4J_URL}")
try:
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("CONNECTION_OK")
    sys.exit(0)
except Exception as e:
    print("CONNECTION_ERR", repr(e))
    sys.exit(1)
