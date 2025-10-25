# DeepSeek Knowledge Graph — README

This deliverable contains a **Neo4j knowledge graph schema** and sample data for the DeepSeek project:
- Cameras, Detections, StateMachines, Events, Alerts, Models, TeamMembers, RAG documents, Tech stack.

Files included:
- deepseek_kg_full.cypher  : Cypher script to create schema and sample data.
- cameras_full.csv         : Camera import template.
- detections_full.csv      : Detection import template.
- state_machine_integration.py : Example module to integrate YOLO state machine with Neo4j.
- deepseek_readme.md       : (this file)

Quick start:
1. Start Neo4j (Docker recommended for testing):
   docker run --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j:latest

2. Copy `deepseek_kg_full.cypher` to the Neo4j host and run:
   cypher-shell -u neo4j -p password -f deepseek_kg_full.cypher

3. Integrate `state_machine_integration.py` into your YOLO pipeline to automatically create Events when state machine thresholds are met.

Notes:
- Times are stored as neo4j datetime (ISO).
- For production, tune indexes, batch writes, and consider archiving raw frame data in object storage rather than KG.
