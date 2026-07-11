from ultralytics import YOLO
import cv2
import time
import math
import numpy as np
from datetime import datetime
from collections import deque
from drone_registry import DroneRegistry
import serial

# ============================================================================
# PROFESSIONAL DRONE SURVEILLANCE SYSTEM WITH ADVANCED MOTION ANALYTICS
# Enterprise Edition v2.0
# ============================================================================

# Configuration
class Config:
    # Camera parameters
    KNOWN_WIDTH = 0.30  # meters
    FOCAL_LENGTH = 700
    CAMERA_FOV = 60
    
    # Display
    WIDTH = 1280
    HEIGHT = 720
    
    # Detection thresholds
    HIGH_THREAT_DISTANCE = 3.0  # meters
    MEDIUM_THREAT_DISTANCE = 7.0
    SWARM_THRESHOLD = 5
    
    # Performance
    SCAN_DURATION = 3  # seconds
    SPEED_SMOOTHING_FRAMES = 10
    DIRECTION_THRESHOLD = 5  # pixels
    TRAJECTORY_HISTORY = 20  # frames
    PREDICTION_FRAMES = 15   # frames ahead
    
    # Speed thresholds (m/s)
    SPEED_SLOW = 2.0
    SPEED_NORMAL = 5.0
    SPEED_FAST = 10.0
    
    # File paths
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    TRACKER_CONFIG = "bytetrack.yaml"

# Professional color palette
class Colors:
    PRIMARY = (0, 120, 255)
    SECONDARY = (80, 200, 200)
    ACCENT = (0, 200, 255)
    SUCCESS = (0, 200, 0)
    WARNING = (0, 165, 255)
    DANGER = (0, 50, 255)
    INFO = (255, 200, 100)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_GRAY = (35, 35, 35)
    MEDIUM_GRAY = (80, 80, 80)
    LIGHT_GRAY = (160, 160, 160)
    PURPLE = (200, 100, 255)
    CYAN = (255, 255, 0)

class DroneAnalytics:
    """Advanced analytics for each drone"""
    def __init__(self, drone_id):
        self.drone_id = drone_id
        self.position_history = deque(maxlen=Config.TRAJECTORY_HISTORY)
        self.speed_history = deque(maxlen=Config.SPEED_SMOOTHING_FRAMES)
        self.velocity_history = deque(maxlen=Config.SPEED_SMOOTHING_FRAMES)
        self.timestamp_history = deque(maxlen=Config.TRAJECTORY_HISTORY)
        
        # Motion metrics
        self.current_speed = 0.0
        self.avg_speed = 0.0
        self.max_speed = 0.0
        self.current_heading = 0.0  # degrees
        self.acceleration = 0.0
        self.trajectory_angle = 0.0
        
        # Position
        self.current_position = (0, 0)
        self.last_position = (0, 0)
        self.last_update = time.time()
        
        # Prediction
        self.predicted_positions = []
        
    def update(self, x, y, distance_m, timestamp, frame_width, frame_height):
        """Update analytics with new position data"""
        self.current_position = (x, y)
        
        if len(self.position_history) > 0:
            last_x, last_y = self.position_history[-1]
            dx = x - last_x
            dy = y - last_y
            dt = timestamp - self.last_update
            
            if dt > 0.033:  # >30fps
                # Calculate actual speed in m/s
                scene_width = 2 * distance_m * math.tan(math.radians(Config.CAMERA_FOV/2))
                meters_per_pixel = scene_width / frame_width
                pixel_distance = math.sqrt(dx**2 + dy**2)
                real_distance = pixel_distance * meters_per_pixel
                speed = real_distance / dt
                
                # Update speed metrics
                self.speed_history.append(speed)
                self.current_speed = sum(self.speed_history) / len(self.speed_history)
                self.avg_speed = sum(self.speed_history) / len(self.speed_history)
                self.max_speed = max(self.max_speed, self.current_speed)
                
                # Calculate velocity vector
                velocity_x = (dx * meters_per_pixel) / dt
                velocity_y = (dy * meters_per_pixel) / dt
                self.velocity_history.append((velocity_x, velocity_y))
                
                # Calculate acceleration
                if len(self.velocity_history) > 1:
                    prev_vx, prev_vy = self.velocity_history[-2]
                    accel_x = (velocity_x - prev_vx) / dt
                    accel_y = (velocity_y - prev_vy) / dt
                    self.acceleration = math.sqrt(accel_x**2 + accel_y**2)
                
                # Calculate heading in degrees
                self.current_heading = math.degrees(math.atan2(dy, dx))
                if self.current_heading < 0:
                    self.current_heading += 360
                
                # Predict future position
                self.predicted_positions = []
                for t in range(1, Config.PREDICTION_FRAMES + 1):
                    pred_x = x + velocity_x * t * dt * 10
                    pred_y = y + velocity_y * t * dt * 10
                    if 0 <= pred_x <= frame_width and 0 <= pred_y <= frame_height:
                        self.predicted_positions.append((int(pred_x), int(pred_y)))
        
        self.position_history.append((x, y))
        self.timestamp_history.append(timestamp)
        if len(self.position_history) > 1:
            self.last_position = self.position_history[-2]
        else:
            self.last_position = (x, y)
        self.last_update = timestamp
        
        # Calculate trajectory angle (overall direction)
        if len(self.position_history) > 5:
            start = self.position_history[0]
            end = self.position_history[-1]
            dx_total = end[0] - start[0]
            dy_total = end[1] - start[1]
            self.trajectory_angle = math.degrees(math.atan2(dy_total, dx_total))
    
    def get_speed_category(self):
        """Get speed category based on current speed"""
        if self.current_speed < Config.SPEED_SLOW:
            return "SLOW", Colors.SUCCESS
        elif self.current_speed < Config.SPEED_NORMAL:
            return "NORMAL", Colors.INFO
        elif self.current_speed < Config.SPEED_FAST:
            return "FAST", Colors.WARNING
        else:
            return "EXTREME", Colors.DANGER
    
    def get_heading_text(self):
        """Convert heading degrees to cardinal direction"""
        headings = [
            (0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
            (180, "S"), (225, "SW"), (270, "W"), (315, "NW"), (360, "N")
        ]
        for angle, direction in headings:
            if self.current_heading <= angle + 22.5:
                return direction
        return "N"
    
    def draw_trajectory(self, frame, color):
        """Draw flight trajectory and prediction"""
        if len(self.position_history) < 2:
            return frame
        
        # Draw trajectory trail
        points = list(self.position_history)
        for i in range(1, len(points)):
            alpha = i / len(points)
            trail_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            cv2.line(frame, points[i-1], points[i], trail_color, 2)
        
        # Draw prediction path
        if self.predicted_positions and len(self.predicted_positions) > 1:
            for i in range(1, len(self.predicted_positions)):
                cv2.line(frame, self.predicted_positions[i-1], self.predicted_positions[i], Colors.PURPLE, 1, cv2.LINE_AA)
            
            # Draw prediction endpoint marker
            last_pred = self.predicted_positions[-1]
            cv2.circle(frame, last_pred, 5, Colors.PURPLE, -1)
            cv2.putText(frame, "PREDICTED", (last_pred[0] - 30, last_pred[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.PURPLE, 1)
        
        return frame
    
    def get_stats_panel(self):
        """Get formatted statistics panel text"""
        speed_cat, speed_color = self.get_speed_category()
        heading_text = self.get_heading_text()
        
        return {
            'speed': f"{self.current_speed:.1f}",
            'speed_cat': speed_cat,
            'speed_color': speed_color,
            'avg_speed': f"{self.avg_speed:.1f}",
            'max_speed': f"{self.max_speed:.1f}",
            'heading': f"{heading_text} ({int(self.current_heading)}°)",
            'accel': f"{self.acceleration:.2f}",
            'traj': f"{int(self.trajectory_angle)}°"
        }

# Initialize
config = Config()
colors = Colors()

model = YOLO(config.MODEL_PATH)
#cap = cv2.VideoCapture('http://10.67.10.35')
cap = cv2.VideoCapture("0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)

# Tracking storage
tracking_data = {}  # track_id -> analytics object
registry = DroneRegistry()

# Session metadata
start_time = time.time()
known_track_ids = set()
alert_message = ""
alert_timestamp = 0
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 60)
print("PROFESSIONAL DRONE SURVEILLANCE SYSTEM v2.0")
print("Advanced Motion Analytics Enabled")
print(f"Session ID: {session_id}")
print(f"Model: {config.MODEL_PATH}")
print(f"Resolution: {config.WIDTH}x{config.HEIGHT}")
print("=" * 60)

def draw_status_bar(frame, drones_count, fps, threat_level, total_speed):
    """Draw top status bar with analytics"""
    bar_height = 60
    cv2.rectangle(frame, (0, 0), (config.WIDTH, bar_height), colors.DARK_GRAY, -1)
    cv2.line(frame, (0, bar_height), (config.WIDTH, bar_height), colors.PRIMARY, 2)
    
    # System name
    cv2.putText(frame, "DRONE SURVEILLANCE SYSTEM", (20, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors.WHITE, 1)
    cv2.putText(frame, "Advanced Motion Analytics", (20, 48), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.LIGHT_GRAY, 1)
    
    # Target counter
    cv2.rectangle(frame, (config.WIDTH - 280, 12), (config.WIDTH - 200, 48), colors.BLACK, -1)
    cv2.rectangle(frame, (config.WIDTH - 280, 12), (config.WIDTH - 200, 48), colors.PRIMARY, 1)
    cv2.putText(frame, f"{drones_count:02d}", (config.WIDTH - 265, 36), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.PRIMARY, 2)
    cv2.putText(frame, "TARGETS", (config.WIDTH - 195, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.LIGHT_GRAY, 1)
    
    # Speed analytics
    cv2.putText(frame, "TOTAL SPEED", (config.WIDTH - 185, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.LIGHT_GRAY, 1)
    cv2.putText(frame, f"{total_speed:.1f} km/h", (config.WIDTH - 185, 48), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.ACCENT, 1)
    
    # Threat level
    threat_x = config.WIDTH - 90
    threat_colors = {0: colors.SUCCESS, 1: colors.INFO, 2: colors.WARNING, 3: colors.DANGER}
    cv2.circle(frame, (threat_x, 28), 6, threat_colors.get(threat_level, colors.WARNING), -1)
    cv2.putText(frame, ["CLEAR", "LOW", "MEDIUM", "HIGH"][threat_level], (threat_x + 15, 32), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.WHITE, 1)
    
    # FPS
    cv2.putText(frame, f"{fps:.1f} FPS", (config.WIDTH - 70, 48), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.MEDIUM_GRAY, 1)
    
    return frame

def draw_detection_box(frame, x1, y1, x2, y2, center_x, center_y, 
                       drone_id, distance, threat, analytics):
    """Draw professional detection box with motion analytics"""
    
    # Color based on threat and speed
    if threat == "HIGH":
        box_color = colors.DANGER
        bg_color = (0, 0, 60)
    elif threat == "MEDIUM":
        box_color = colors.WARNING
        bg_color = (0, 60, 60)
    else:
        speed_cat, speed_color = analytics.get_speed_category()
        box_color = speed_color if analytics.current_speed > 0 else colors.SUCCESS
        bg_color = (0, 60, 0)
    
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Animated corner brackets
    bracket_len = 20
    # Top-left
    cv2.line(frame, (x1, y1 + bracket_len), (x1, y1), box_color, 2)
    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), box_color, 2)
    # Top-right
    cv2.line(frame, (x2, y1 + bracket_len), (x2, y1), box_color, 2)
    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), box_color, 2)
    # Bottom-left
    cv2.line(frame, (x1, y2 - bracket_len), (x1, y2), box_color, 2)
    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), box_color, 2)
    # Bottom-right
    cv2.line(frame, (x2, y2 - bracket_len), (x2, y2), box_color, 2)
    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), box_color, 2)
    
    # Draw trajectory
    frame = analytics.draw_trajectory(frame, box_color)
    
    # Main info badge
    badge_width = 220
    badge_height = 85
    badge_x = x1
    badge_y = y1 - badge_height - 5
    
    if badge_y < 5:
        badge_y = y2 + 5
    
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), bg_color, -1)
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), box_color, 1)
    
    # Header - Drone ID and Threat
    cv2.putText(frame, f"[{drone_id}] {threat}", (badge_x + 8, badge_y + 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    # Distance
    cv2.putText(frame, f"DIST: {distance:.1f}m", (badge_x + 8, badge_y + 34), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.LIGHT_GRAY, 1)
    
    # Speed with category
    stats = analytics.get_stats_panel()
    cv2.putText(frame, f"SPD: {stats['speed']} m/s", (badge_x + 8, badge_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, stats['speed_color'], 1)
    cv2.putText(frame, f"[{stats['speed_cat']}]", (badge_x + 110, badge_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, stats['speed_color'], 1)
    
    # Heading and acceleration
    cv2.putText(frame, f"HDG: {stats['heading']}", (badge_x + 8, badge_y + 66), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.CYAN, 1)
    cv2.putText(frame, f"ACC: {stats['accel']} m/s²", (badge_x + 110, badge_y + 66), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors.LIGHT_GRAY, 1)
    
    # Speed indicator bar (0-15 m/s range)
    bar_width = min(int((analytics.current_speed / 15) * 90), 90)
    cv2.rectangle(frame, (badge_x + 125, badge_y + 48), (badge_x + 215, badge_y + 53), colors.MEDIUM_GRAY, -1)
    cv2.rectangle(frame, (badge_x + 125, badge_y + 48), (badge_x + 125 + bar_width, badge_y + 53), stats['speed_color'], -1)
    
    # Crosshair with heading indicator
    cv2.line(frame, (center_x - 15, center_y), (center_x - 5, center_y), box_color, 1)
    cv2.line(frame, (center_x + 5, center_y), (center_x + 15, center_y), box_color, 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y - 5), box_color, 1)
    cv2.line(frame, (center_x, center_y + 5), (center_x, center_y + 15), box_color, 1)
    cv2.circle(frame, (center_x, center_y), 4, box_color, 1)
    
    # Heading direction arrow
    heading_rad = math.radians(analytics.current_heading)
    arrow_end_x = int(center_x + 20 * math.cos(heading_rad))
    arrow_end_y = int(center_y + 20 * math.sin(heading_rad))
    cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y), colors.ACCENT, 2, tipLength=0.3)
    
    return frame

def draw_target_panel(frame, targets):
    """Draw right side target information panel with analytics"""
    if not targets:
        return frame
    
    panel_x = config.WIDTH - 280
    panel_y = 70
    max_height = min(500, 45 + len(targets) * 45)
    
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (config.WIDTH - 10, panel_y + max_height), colors.DARK_GRAY, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (config.WIDTH - 10, panel_y + max_height), colors.PRIMARY, 1)
    
    # Panel header
    cv2.rectangle(frame, (panel_x, panel_y), (config.WIDTH - 10, panel_y + 35), colors.PRIMARY, -1)
    cv2.putText(frame, "TARGET ANALYSIS", (panel_x + 15, panel_y + 24), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.WHITE, 1)
    
    # Sort by distance
    targets.sort(key=lambda x: x['distance'])
    
    y = panel_y + 48
    for i, target in enumerate(targets[:9]):
        # Row background
        row_color = colors.DARK_GRAY if i % 2 == 0 else (45, 45, 45)
        cv2.rectangle(frame, (panel_x + 5, y - 20), (config.WIDTH - 15, y + 18), row_color, -1)
        
        # Threat indicator bar
        threat_color = colors.DANGER if target['threat'] == "HIGH" else colors.WARNING if target['threat'] == "MEDIUM" else colors.SUCCESS
        cv2.rectangle(frame, (panel_x + 5, y - 20), (panel_x + 8, y + 18), threat_color, -1)
        
        # Target ID and distance
        cv2.putText(frame, target['id'], (panel_x + 18, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.WHITE, 1)
        cv2.putText(frame, f"{target['distance']:.1f}m", (panel_x + 100, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.LIGHT_GRAY, 1)
        
        # Speed with unit
        speed_color = colors.SUCCESS if target['speed'] < 2 else colors.WARNING if target['speed'] < 5 else colors.DANGER
        cv2.putText(frame, f"{target['speed']:.1f}m/s", (panel_x + 160, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, speed_color, 1)
        
        # Heading arrow
        cv2.putText(frame, target['heading_arrow'], (panel_x + 225, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.ACCENT, 1)
        
        # Acceleration indicator
        accel_indicator = "↑" if target['acceleration'] > 0.5 else "→" if target['acceleration'] > 0.1 else "↓"
        cv2.putText(frame, accel_indicator, (panel_x + 248, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.INFO, 1)
        
        y += 42
    
    if len(targets) > 9:
        cv2.putText(frame, f"+ {len(targets) - 9} more", (panel_x + 15, y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.MEDIUM_GRAY, 1)
    
    return frame

def draw_motion_graph(frame, analytics):
    """Draw real-time motion graph for selected drone"""
    if not analytics or len(analytics.speed_history) < 2:
        return frame
    
    graph_x = 20
    graph_y = config.HEIGHT - 120
    graph_w = 300
    graph_h = 80
    
    # Graph background
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), colors.DARK_GRAY, -1)
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), colors.MEDIUM_GRAY, 1)
    
    # Title
    cv2.putText(frame, "SPEED HISTORY (m/s)", (graph_x + 5, graph_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.LIGHT_GRAY, 1)
    
    # Plot speed history
    speeds = list(analytics.speed_history)
    max_speed = max(speeds + [10])
    
    points = []
    for i, speed in enumerate(speeds):
        x = graph_x + 5 + int(i * (graph_w - 10) / len(speeds))
        y = graph_y + graph_h - 10 - int((speed / max_speed) * (graph_h - 20))
        points.append((x, y))
    
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], colors.ACCENT, 2)
    
    # Current speed line
    current_y = graph_y + graph_h - 10 - int((analytics.current_speed / max_speed) * (graph_h - 20))
    cv2.line(frame, (graph_x + 5, current_y), (graph_x + graph_w - 5, current_y), colors.WARNING, 1)
    
    return frame

# Initialize FPS calculation
fps = 0
fps_timer = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
    
    # Detection and tracking
    results = model.track(frame, persist=True, tracker=config.TRACKER_CONFIG, verbose=False)
    annotated = results[0].plot()
    
    targets = []
    active_drones = 0
    total_speed = 0
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        active_drones = len(set(track_ids))
        
        # Check for new drones
        current_ids = set(track_ids)
        new_ids = current_ids - known_track_ids
        for nid in new_ids:
            alert_message = f"NEW TARGET ACQUIRED | ID: {nid}"
            alert_timestamp = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] NEW TARGET: ID {nid}")
        known_track_ids = current_ids
        
        for box, track_id in zip(results[0].boxes, track_ids):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
            pixel_width = x2 - x1
            
            if pixel_width <= 0:
                continue
            
            # Calculate metrics
            distance = (config.KNOWN_WIDTH * config.FOCAL_LENGTH) / pixel_width
            permanent_id = registry.get_drone_id(track_id, center_x, center_y, x2-x1, y2-y1)
            
            # Initialize or update analytics
            if track_id not in tracking_data:
                tracking_data[track_id] = DroneAnalytics(permanent_id)
            
            analytics = tracking_data[track_id]
            current_time = time.time()
            analytics.update(center_x, center_y, distance, current_time, config.WIDTH, config.HEIGHT)
            
            # Threat assessment
            if distance < config.HIGH_THREAT_DISTANCE:
                threat = "HIGH"
            elif distance < config.MEDIUM_THREAT_DISTANCE:
                threat = "MEDIUM"
            else:
                threat = "LOW"
            
            # Draw detection with analytics
            annotated = draw_detection_box(annotated, x1, y1, x2, y2, center_x, center_y,
                                           permanent_id, distance, threat, analytics)
            
            total_speed += analytics.current_speed
            
            stats = analytics.get_stats_panel()
            targets.append({
                'id': permanent_id,
                'distance': distance,
                'threat': threat,
                'speed': analytics.current_speed,
                'heading': analytics.get_heading_text(),
                'heading_arrow': analytics.get_heading_text()[0] if analytics.get_heading_text() else "N",
                'acceleration': analytics.acceleration
            })
    
    # Calculate FPS
    frame_count += 1
    if time.time() - fps_timer >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_timer = time.time()
    
    # Determine threat level
    threat_level = 0
    if active_drones >= config.SWARM_THRESHOLD:
        threat_level = 3
    elif any(t['threat'] == "HIGH" for t in targets):
        threat_level = 3
    elif any(t['threat'] == "MEDIUM" for t in targets):
        threat_level = 2
    elif active_drones > 0:
        threat_level = 1
    
    # Draw UI components
    annotated = draw_status_bar(annotated, active_drones, fps, threat_level, total_speed)
    annotated = draw_target_panel(annotated, targets)
    
    # Draw motion graph for nearest target
    if targets and tracking_data:
        nearest_track_id = list(tracking_data.keys())[0]  # Show first drone's graph
        annotated = draw_motion_graph(annotated, tracking_data[nearest_track_id])
    
    # Show alerts
    if time.time() - alert_timestamp < 3.0:
        alert_type = "DANGER" if threat_level >= 3 else "WARNING" if threat_level >= 2 else "INFO"
        alert_color = colors.DANGER if alert_type == "DANGER" else colors.WARNING if alert_type == "WARNING" else colors.INFO
        cv2.rectangle(annotated, (config.WIDTH//2 - 200, config.HEIGHT - 110), 
                     (config.WIDTH//2 + 200, config.HEIGHT - 75), colors.BLACK, -1)
        cv2.rectangle(annotated, (config.WIDTH//2 - 200, config.HEIGHT - 110), 
                     (config.WIDTH//2 + 200, config.HEIGHT - 75), alert_color, 2)
        cv2.putText(annotated, alert_message, (config.WIDTH//2 - 180, config.HEIGHT - 87), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
    
    # Swarm warning
    if active_drones >= config.SWARM_THRESHOLD:
        cv2.rectangle(annotated, (0, config.HEIGHT-100), (config.WIDTH, config.HEIGHT-70), colors.DANGER, -1)
        cv2.putText(annotated, "SWARM DETECTION PROTOCOL ACTIVE", (config.WIDTH//2 - 200, config.HEIGHT-82), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.WHITE, 1)
    
    # Footer
    footer_y = config.HEIGHT - 28
    cv2.rectangle(annotated, (0, footer_y), (config.WIDTH, config.HEIGHT), colors.DARK_GRAY, -1)
    cv2.line(annotated, (0, footer_y), (config.WIDTH, footer_y), colors.MEDIUM_GRAY, 1)
    
    commands = [("ESC", "EXIT"), ("R", "RESET"), ("S", "SCREENSHOT")]
    x = 20
    for cmd, desc in commands:
        cv2.putText(annotated, cmd, (x, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.PRIMARY, 1)
        cv2.putText(annotated, desc, (x + 45, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.MEDIUM_GRAY, 1)
        x += 130
    
    time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(annotated, time_str, (config.WIDTH - 80, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.MEDIUM_GRAY, 1)
    
    # Display
    cv2.imshow("Drone Surveillance System - Motion Analytics", annotated)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] System shutdown")
        break
    elif key == ord('r') or key == ord('R'):
        tracking_data = {}
        registry = DroneRegistry()
        known_track_ids = set()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] System reset")
        alert_message = "System reset complete"
        alert_timestamp = time.time()
    elif key == ord('s') or key == ord('S'):
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, annotated)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Screenshot saved: {filename}")
        alert_message = "Screenshot captured"
        alert_timestamp = time.time()

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("=" * 60)
print("SYSTEM SHUTDOWN COMPLETE")
print(f"Session duration: {int(time.time() - start_time)} seconds")
print("=" * 60)