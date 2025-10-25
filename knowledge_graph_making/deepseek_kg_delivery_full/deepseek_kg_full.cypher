// deepseek_kg_full.cypher
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
MERGE (t1:TeamMember {name:'李康', role:'Backend Engineer', uni:'河南信阳学院', year:2, skills:['python','flask','neo4j']})
MERGE (t2:TeamMember {name:'朱梓华', role:'ML Engineer', uni:'河南信阳学院', year:2, skills:['pytorch','yolo','quantization','LoRA']})
MERGE (t3:TeamMember {name:'杜明阳', role:'Frontend Engineer', uni:'河南信阳学院', year:2, skills:['javascript','react','canvas']})
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
