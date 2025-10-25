# Retry: Create expanded DeepSeek KG deliverable files in /mnt/data
from pathlib import Path
out_dir = Path(r"D:\PyCharm 2025.1.1.1\DS_tuning\knowledge_graph_making\deepseek_kg_delivery_full")
out_dir.mkdir(parents=True, exist_ok=True)

cypher = """// deepseek_kg_full.cypher
// Full DeepSeek Knowledge Graph schema + sample data
// Run in Neo4j Browser or via cypher-shell (ensure NEO4J_AUTH is set)

CREATE CONSTRAINT IF NOT EXISTS FOR (c:Camera) REQUIRE c.camera_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Detection) REQUIRE d.detection_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (x:Detection) ON (x.timestamp);
CREATE INDEX IF NOT EXISTS FOR (x:Event) ON (x.start_time);

// Project node
MERGE (proj:Project {name:'DeepSeek', short:'DeepSeek Fall Detection & Alerting', created:date(), description:'实时摄像头摔倒检测 + DS 大模型告警与 RAG 记忆层的系统'})

// Team members
MERGE (t1:TeamMember {name:'李明', role:'Backend Engineer', uni:'河南科技大学', year:2, skills:['python','flask','neo4j']})
MERGE (t2:TeamMember {name:'王芳', role:'ML Engineer', uni:'河南工程学院', year:2, skills:['pytorch','yolo','quantization','LoRA']})
MERGE (t3:TeamMember {name:'赵强', role:'Frontend Engineer', uni:'河南师范大学', year:2, skills:['javascript','react','canvas']})
MERGE (t1)-[:WORKS_ON]->(proj)
MERGE (t2)-[:WORKS_ON]->(proj)
MERGE (t3)-[:WORKS_ON]->(proj)

// Models
MERGE (m_yolo:Model {name:'YOLOvX-Fall', version:'1.0', type:'detector', description:'训练标签: Fall Detected, Normal, Resting', trained_on:'custom dataset', notes:'用于实时检测'})
MERGE (m_ds:Model {name:'DS7b', version:'4bit-LoRA-v1', type:'assistant', quantization:'4bit', finetune:'LoRA', description:'用于告警生成与问答', notes:'部署为本地 REST 服务'})

// Cameras & Locations
MERGE (loc1:Location {site:'Building A', floor:'2', area:'Staircase'})
MERGE (loc2:Location {site:'Building A', floor:'1', area:'Lobby'})
MERGE (cam1:Camera {camera_id:'cam_001', ip:'rtsp://10.0.0.5/stream', location:'Building A - Floor 2 - Staircase', description:'Staircase camera #1', installed:date('2025-10-01')})
MERGE (cam2:Camera {camera_id:'cam_002', ip:'rtsp://10.0.0.6/stream', location:'Building A - Floor 1 - Lobby', description:'Lobby camera #1', installed:date('2025-09-20')})
MERGE (cam1)-[:INSTALLED_AT]->(loc1)
MERGE (cam2)-[:INSTALLED_AT]->(loc2)
MERGE (proj)-[:HAS_CAMERA]->(cam1)
MERGE (proj)-[:HAS_CAMERA]->(cam2)

// Sample Detections (more rows)
CREATE (d1:Detection {detection_id:'det_cam001_0001', timestamp:datetime('2025-10-23T22:00:12Z'), label:'Fall Detected', confidence:0.93, frame_id:3456})
CREATE (d2:Detection {detection_id:'det_cam001_0002', timestamp:datetime('2025-10-23T22:00:14Z'), label:'Fall Detected', confidence:0.90, frame_id:3458})
CREATE (d3:Detection {detection_id:'det_cam001_0003', timestamp:datetime('2025-10-23T22:00:20Z'), label:'Normal', confidence:0.88, frame_id:3464})
CREATE (d4:Detection {detection_id:'det_cam002_0001', timestamp:datetime('2025-10-24T09:15:05Z'), label:'Resting', confidence:0.85, frame_id:1001})
CREATE (d5:Detection {detection_id:'det_cam002_0002', timestamp:datetime('2025-10-24T09:15:10Z'), label:'Fall Detected', confidence:0.88, frame_id:1004})
CREATE (d6:Detection {detection_id:'det_cam002_0003', timestamp:datetime('2025-10-24T09:15:20Z'), label:'Fall Detected', confidence:0.91, frame_id:1008})

// Link detections to cameras
MATCH (c1:Camera {camera_id:'cam_001'}), (c2:Camera {camera_id:'cam_002'})
MATCH (d1:Detection {detection_id:'det_cam001_0001'}), (d2:Detection {detection_id:'det_cam001_0002'}), (d3:Detection {detection_id:'det_cam001_0003'}), (d4:Detection {detection_id:'det_cam002_0001'}), (d5:Detection {detection_id:'det_cam002_0002'}), (d6:Detection {detection_id:'det_cam002_0003'})
MERGE (c1)-[:HAS_DETECTION]->(d1)
MERGE (c1)-[:HAS_DETECTION]->(d2)
MERGE (c1)-[:HAS_DETECTION]->(d3)
MERGE (c2)-[:HAS_DETECTION]->(d4)
MERGE (c2)-[:HAS_DETECTION]->(d5)
MERGE (c2)-[:HAS_DETECTION]->(d6)

// Sample StateMachine / Event creation (simulate a 10s fall event)
CREATE (sm1:StateMachine {sm_id:'sm_1001', camera_id:'cam_001', status:'triggered', started:datetime('2025-10-23T22:00:12Z'), last_update:datetime('2025-10-23T22:00:22Z')})
CREATE (e1:Event {event_id:'evt_0001', type:'Fall', start_time:datetime('2025-10-23T22:00:12Z'), end_time:datetime('2025-10-23T22:00:22Z'), duration_sec:10, severity:'high', summary:'Staircase fall persisted 10s'})
MERGE (cam1)-[:GENERATED]->(e1)
MERGE (sm1)-[:TRIGGERS]->(e1)
MERGE (e1)-[:COMPOSED_OF]->(d1)
MERGE (e1)-[:COMPOSED_OF]->(d2)

// Another Event example
CREATE (sm2:StateMachine {sm_id:'sm_2001', camera_id:'cam_002', status:'triggered', started:datetime('2025-10-24T09:15:10Z'), last_update:datetime('2025-10-24T09:15:25Z')})
CREATE (e2:Event {event_id:'evt_0002', type:'Fall', start_time:datetime('2025-10-24T09:15:10Z'), end_time:datetime('2025-10-24T09:15:25Z'), duration_sec:15, severity:'high', summary:'Lobby fall persisted 15s'})
MERGE (cam2)-[:GENERATED]->(e2)
MERGE (sm2)-[:TRIGGERS]->(e2)
MERGE (e2)-[:COMPOSED_OF]->(d5)
MERGE (e2)-[:COMPOSED_OF]->(d6)

// Alerts (DS model outputs)
CREATE (a1:Alert {alert_id:'alert_0001', timestamp:datetime('2025-10-23T22:00:24Z'), message:'[告警] 2025-10-23 22:00 cam_001: 检测到摔倒，持续 10 秒', severity:'high', ds_model:'DS7b-4bit-LoRA-v1'})
CREATE (a2:Alert {alert_id:'alert_0002', timestamp:datetime('2025-10-24T09:15:27Z'), message:'[告警] 2025-10-24 09:15 cam_002: 检测到摔倒，持续 15 秒', severity:'high', ds_model:'DS7b-4bit-LoRA-v1'})
MERGE (e1)-[:REPORTED_AS]->(a1)
MERGE (e2)-[:REPORTED_AS]->(a2)

// RAG memory documents (short)
CREATE (doc1:RAGDoc {doc_id:'doc_001', title:'Event evt_0001 transcript', content:'Staircase fall at 2025-10-23 22:00. Duration 10s. Confidence high.', created:datetime('2025-10-23T22:00:30Z')})
CREATE (doc2:RAGDoc {doc_id:'doc_002', title:'Event evt_0002 transcript', content:'Lobby fall at 2025-10-24 09:15. Duration 15s. Confidence high.', created:datetime('2025-10-24T09:15:30Z')})
MERGE (e1)-[:HAS_DOC]->(doc1)
MERGE (e2)-[:HAS_DOC]->(doc2)

// Tech stack
MERGE (tech1:Tech {name:'OpenCV', role:'video capture & preprocessing'})
MERGE (tech2:Tech {name:'YOLO', role:'object detection (custom-trained labels)'})
MERGE (tech3:Tech {name:'PyTorch', role:'model training & inference'})
MERGE (tech4:Tech {name:'Neo4j', role:'knowledge graph & storage'})
MERGE (tech5:Tech {name:'FastAPI', role:'model & KG API'})
MERGE (proj)-[:USES]->(tech1)
MERGE (proj)-[:USES]->(tech2)
MERGE (proj)-[:USES]->(tech3)
MERGE (proj)-[:USES]->(tech4)
MERGE (proj)-[:USES]->(tech5)

// Model-analysis relationships
MERGE (m_yolo)-[:DETECTS]->(d1)
MERGE (m_yolo)-[:DETECTS]->(d2)
MERGE (m_ds)-[:ANALYZED]->(e1)
MERGE (m_ds)-[:ANALYZED]->(e2)

// Return info
RETURN 'DeepSeek full KG loaded' AS info;
"""

readme = """# DeepSeek Knowledge Graph — README

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
"""

cameras_csv = "camera_id,ip,location,description,installed\ncam_001,rtsp://10.0.0.5/stream,Building A - Floor 2 - Staircase,Staircase camera #1,2025-10-01\ncam_002,rtsp://10.0.0.6/stream,Building A - Floor 1 - Lobby,Lobby camera #1,2025-09-20\n"

detections_csv = "detection_id,timestamp,label,confidence,frame_id,camera_id\n"
rows = [
    ("det_cam001_0001","2025-10-23T22:00:12Z","Fall Detected",0.93,3456,"cam_001"),
    ("det_cam001_0002","2025-10-23T22:00:14Z","Fall Detected",0.90,3458,"cam_001"),
    ("det_cam001_0003","2025-10-23T22:00:20Z","Normal",0.88,3464,"cam_001"),
    ("det_cam002_0001","2025-10-24T09:15:05Z","Resting",0.85,1001,"cam_002"),
    ("det_cam002_0002","2025-10-24T09:15:10Z","Fall Detected",0.88,1004,"cam_002"),
    ("det_cam002_0003","2025-10-24T09:15:20Z","Fall Detected",0.91,1008,"cam_002")
]
for r in rows:
    detections_csv += ",".join([str(x) for x in r]) + "\n"

state_machine_py = """# state_machine_integration.py
# Example integration of YOLO detections + simple state machine to trigger Neo4j Event creation.
# This module demonstrates logic; adapt to your project's threading and async model.

from datetime import datetime, timezone
from collections import deque
from neo4j import GraphDatabase
import uuid
import os

NEO4J_URI = os.getenv('NEO4J_URI','bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER','neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD','password')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class SimpleStateMachine:
    def __init__(self, camera_id, threshold_seconds=10, fps=5):
        self.camera_id = camera_id
        self.threshold = threshold_seconds
        self.fps = fps
        self.buffer = deque()  # store tuples (timestamp, label, confidence, frame_id)
        self.triggered = False
        self.start_time = None

    def feed_detection(self, label, confidence, frame_id, ts=None):
        # ts expected as ISO string or datetime
        if ts is None:
            ts = datetime.now(timezone.utc).isoformat()
        self.buffer.append({'detection_id': f"{self.camera_id}_{frame_id}", 'timestamp': ts, 'label': label, 'confidence':confidence, 'frame_id':frame_id})
        # maintain buffer size roughly threshold*fps*2
        maxlen = max(50, int(self.threshold * self.fps * 2))
        while len(self.buffer) > maxlen:
            self.buffer.popleft()

        # state logic: simple: if label == 'Fall Detected' then start/continue timer; if Normal/Resting reset
        if label == 'Fall Detected':
            if not self.triggered:
                self.triggered = True
                self.start_time = ts
        elif label in ('Normal','Resting'):
            # reset
            self.triggered = False
            self.start_time = None

        # check duration
        if self.triggered and self.start_time:
            # compute duration in seconds (simple parse)
            try:
                from datetime import datetime
                s = datetime.fromisoformat(self.start_time.replace('Z','+00:00'))
                now = datetime.fromisoformat(ts.replace('Z','+00:00'))
                duration = (now - s).total_seconds()
                if duration >= self.threshold:
                    # create event
                    detections = list(self.buffer)[-int(self.threshold*self.fps):]
                    event_id = self.create_event_in_neo4j(detections, self.start_time, ts, int(duration))
                    # after creating, reset machine
                    self.triggered = False
                    self.start_time = None
                    return event_id
            except Exception as e:
                print('duration calc error', e)
        return None

    def create_event_in_neo4j(self, detections, start_ts, end_ts, duration_sec):
        event_id = 'evt_' + str(uuid.uuid4())[:8]
        with driver.session() as session:
            session.write_transaction(self._tx_create_event, self.camera_id, detections, event_id, start_ts, end_ts, duration_sec)
        return event_id

    @staticmethod
    def _tx_create_event(tx, camera_id, detections, event_id, start_ts, end_ts, duration_sec):
        tx.run('''MATCH (c:Camera {camera_id:$camera_id})
        CREATE (e:Event {
            event_id:$event_id,
            type:'Fall',
            start_time:datetime($start_ts),
            end_time:datetime($end_ts),
            duration_sec:$duration_sec,
            severity:'high',
            summary: 'Auto-created from state machine'
        })
        CREATE (c)-[:GENERATED]->(e)
        WITH e
        UNWIND $detections AS det
          MERGE (d:Detection {detection_id:det.detection_id})
          SET d.timestamp = datetime(det.timestamp), d.label = det.label, d.confidence = det.confidence, d.frame_id = det.frame_id
          CREATE (e)-[:COMPOSED_OF]->(d)
        RETURN e.event_id
        ''', camera_id=camera_id, detections=detections, event_id=event_id, start_ts=start_ts, end_ts=end_ts, duration_sec=duration_sec)

"""

# write files to out_dir
file_contents = {
    'deepseek_kg_full.cypher': cypher,
    'deepseek_readme.md': readme,
    'cameras_full.csv': cameras_csv,
    'detections_full.csv': detections_csv,
    'state_machine_integration.py': state_machine_py
}

for fname, content in file_contents.items():
    p = out_dir / fname
    p.write_text(content, encoding='utf-8')

# print report
for fname in file_contents:
    p = out_dir / fname
    print(f"Created: {p} ({p.stat().st_size} bytes) -> sandbox:{p}")

