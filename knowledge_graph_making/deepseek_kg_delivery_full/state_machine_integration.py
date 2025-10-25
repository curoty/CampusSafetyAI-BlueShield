# state_machine_integration.py
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

