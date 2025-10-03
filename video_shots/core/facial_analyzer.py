"""Advanced facial analysis for detecting micro-expressions and changes."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class FacialState(Enum):
    """Facial states that can be detected."""
    EYES_OPEN = "eyes_open"
    EYES_CLOSED = "eyes_closed"
    EYES_SQUINTING = "eyes_squinting"
    MOUTH_CLOSED = "mouth_closed"
    MOUTH_OPEN = "mouth_open"
    MOUTH_SMILING = "mouth_smiling"
    EYEBROWS_RAISED = "eyebrows_raised"
    EYEBROWS_FURROWED = "eyebrows_furrowed"


@dataclass
class FacialFeatures:
    """Comprehensive facial features."""
    landmarks: Optional[np.ndarray]
    eye_aspect_ratio_left: float
    eye_aspect_ratio_right: float
    mouth_aspect_ratio: float
    eyebrow_height: float
    face_orientation: Tuple[float, float, float]  # yaw, pitch, roll
    facial_states: List[FacialState]
    confidence: float
    face_bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class FacialChange:
    """Detected facial change between frames."""
    change_type: str
    severity: float  # 0.0 to 1.0
    description: str
    confidence: float
    previous_state: List[FacialState]
    current_state: List[FacialState]


class FacialAnalyzer:
    """Advanced facial analysis for detecting micro-expressions and changes."""
    
    def __init__(self, use_dlib: bool = True):
        """
        Initialize facial analyzer.
        
        Args:
            use_dlib: Whether to use dlib for more accurate landmark detection
        """
        self.use_dlib = use_dlib and DLIB_AVAILABLE
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Initialize dlib if available
        if self.use_dlib:
            try:
                self.detector = dlib.get_frontal_face_detector()
                # Note: You would need to download shape_predictor_68_face_landmarks.dat
                # and place it in the appropriate location
                self.predictor = None  # dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            except:
                self.use_dlib = False
        
        # Eye aspect ratio thresholds
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 3
        
        # Mouth aspect ratio thresholds
        self.MAR_THRESHOLD = 0.5
        
        # Previous frame data for comparison
        self.previous_features: Optional[FacialFeatures] = None
    
    def analyze_face(self, frame: np.ndarray) -> Optional[FacialFeatures]:
        """
        Analyze facial features in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            FacialFeatures object or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract facial features
        if self.use_dlib and self.predictor:
            landmarks = self._get_dlib_landmarks(gray, face)
        else:
            landmarks = self._get_opencv_landmarks(gray, face)
        
        if landmarks is None:
            return None
        
        # Calculate facial metrics
        ear_left = self._calculate_eye_aspect_ratio(landmarks, 'left')
        ear_right = self._calculate_eye_aspect_ratio(landmarks, 'right')
        mar = self._calculate_mouth_aspect_ratio(landmarks)
        eyebrow_height = self._calculate_eyebrow_height(landmarks)
        orientation = self._estimate_head_orientation(landmarks, (w, h))
        
        # Determine facial states
        states = self._determine_facial_states(ear_left, ear_right, mar, eyebrow_height)
        
        # Calculate confidence based on face size and detection quality
        confidence = min(1.0, (w * h) / (frame.shape[0] * frame.shape[1] * 0.1))
        
        return FacialFeatures(
            landmarks=landmarks,
            eye_aspect_ratio_left=ear_left,
            eye_aspect_ratio_right=ear_right,
            mouth_aspect_ratio=mar,
            eyebrow_height=eyebrow_height,
            face_orientation=orientation,
            facial_states=states,
            confidence=confidence,
            face_bbox=(x, y, w, h)
        )
    
    def detect_facial_changes(self, current_features: FacialFeatures) -> List[FacialChange]:
        """
        Detect changes between current and previous facial features.
        
        Args:
            current_features: Current frame facial features
            
        Returns:
            List of detected facial changes
        """
        if self.previous_features is None:
            self.previous_features = current_features
            return []
        
        changes = []
        prev = self.previous_features
        curr = current_features
        
        # Eye state changes
        eye_change = self._detect_eye_changes(prev, curr)
        if eye_change:
            changes.append(eye_change)
        
        # Mouth state changes
        mouth_change = self._detect_mouth_changes(prev, curr)
        if mouth_change:
            changes.append(mouth_change)
        
        # Eyebrow changes
        eyebrow_change = self._detect_eyebrow_changes(prev, curr)
        if eyebrow_change:
            changes.append(eyebrow_change)
        
        # Head orientation changes
        orientation_change = self._detect_orientation_changes(prev, curr)
        if orientation_change:
            changes.append(orientation_change)
        
        self.previous_features = current_features
        return changes
    
    def _get_dlib_landmarks(self, gray: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get facial landmarks using dlib (more accurate)."""
        if not self.predictor:
            return self._get_opencv_landmarks(gray, face)
        
        x, y, w, h = face
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(gray, rect)
        
        # Convert to numpy array
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        return points
    
    def _get_opencv_landmarks(self, gray: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get simplified facial landmarks using OpenCV."""
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes within face region
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        if len(eyes) < 2:
            # Generate simplified landmarks based on face geometry
            return self._generate_geometric_landmarks(face)
        
        # Sort eyes by x coordinate (left, right)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Create simplified landmark set
        landmarks = []
        
        # Eye landmarks (simplified)
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            # Eye center
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            
            # Eye corners (estimated)
            landmarks.extend([
                (eye_center_x - ew//2, eye_center_y),  # Left corner
                (eye_center_x, eye_center_y - eh//2),  # Top
                (eye_center_x + ew//2, eye_center_y),  # Right corner
                (eye_center_x, eye_center_y + eh//2),  # Bottom
            ])
        
        # Mouth landmarks (estimated)
        mouth_y = y + int(h * 0.75)
        mouth_x = x + w // 2
        mouth_w = w // 3
        landmarks.extend([
            (mouth_x - mouth_w//2, mouth_y),  # Left corner
            (mouth_x, mouth_y - 5),           # Top
            (mouth_x + mouth_w//2, mouth_y),  # Right corner
            (mouth_x, mouth_y + 5),           # Bottom
        ])
        
        # Eyebrow landmarks (estimated)
        eyebrow_y = y + int(h * 0.25)
        for i in range(2):
            brow_x = x + int(w * 0.25) + i * int(w * 0.5)
            landmarks.extend([
                (brow_x - 15, eyebrow_y),
                (brow_x, eyebrow_y - 5),
                (brow_x + 15, eyebrow_y)
            ])
        
        return np.array(landmarks) if landmarks else None
    
    def _generate_geometric_landmarks(self, face: Tuple[int, int, int, int]) -> np.ndarray:
        """Generate landmarks based on face geometry when detection fails."""
        x, y, w, h = face
        
        # Standard facial proportions
        landmarks = []
        
        # Eyes (approximately 1/3 from top, 1/5 from sides)
        eye_y = y + h // 3
        left_eye_x = x + w // 5
        right_eye_x = x + 4 * w // 5
        
        # Left eye landmarks
        landmarks.extend([
            (left_eye_x - 10, eye_y),      # Left corner
            (left_eye_x, eye_y - 5),       # Top
            (left_eye_x + 10, eye_y),      # Right corner  
            (left_eye_x, eye_y + 5),       # Bottom
        ])
        
        # Right eye landmarks
        landmarks.extend([
            (right_eye_x - 10, eye_y),     # Left corner
            (right_eye_x, eye_y - 5),      # Top
            (right_eye_x + 10, eye_y),     # Right corner
            (right_eye_x, eye_y + 5),      # Bottom
        ])
        
        # Mouth (approximately 2/3 from top)
        mouth_y = y + 2 * h // 3
        mouth_x = x + w // 2
        landmarks.extend([
            (mouth_x - 15, mouth_y),       # Left corner
            (mouth_x, mouth_y - 5),        # Top
            (mouth_x + 15, mouth_y),       # Right corner
            (mouth_x, mouth_y + 5),        # Bottom
        ])
        
        # Eyebrows
        eyebrow_y = y + h // 4
        landmarks.extend([
            (left_eye_x - 5, eyebrow_y),   # Left eyebrow
            (left_eye_x + 5, eyebrow_y - 3),
            (right_eye_x - 5, eyebrow_y - 3),
            (right_eye_x + 5, eyebrow_y),  # Right eyebrow
        ])
        
        return np.array(landmarks)
    
    def _calculate_eye_aspect_ratio(self, landmarks: np.ndarray, eye: str) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        if len(landmarks) < 8:  # Need at least 8 points for both eyes
            return 0.3  # Default "open" value
        
        if eye == 'left':
            # Use first 4 landmarks for left eye
            eye_points = landmarks[0:4]
        else:
            # Use next 4 landmarks for right eye
            eye_points = landmarks[4:8] if len(landmarks) >= 8 else landmarks[0:4]
        
        if len(eye_points) < 4:
            return 0.3
        
        # Calculate distances
        # Vertical distances
        vertical_1 = self._euclidean_distance(eye_points[1], eye_points[3])
        
        # Horizontal distance  
        horizontal = self._euclidean_distance(eye_points[0], eye_points[2])
        
        # EAR calculation
        ear = vertical_1 / (horizontal + 1e-6)
        return ear
    
    def _calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for speech detection."""
        if len(landmarks) < 12:  # Need mouth landmarks
            return 0.0
        
        # Use landmarks 8-11 for mouth
        mouth_points = landmarks[8:12] if len(landmarks) >= 12 else landmarks[-4:]
        
        if len(mouth_points) < 4:
            return 0.0
        
        # Vertical distance
        vertical = self._euclidean_distance(mouth_points[1], mouth_points[3])
        
        # Horizontal distance
        horizontal = self._euclidean_distance(mouth_points[0], mouth_points[2])
        
        # MAR calculation
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def _calculate_eyebrow_height(self, landmarks: np.ndarray) -> float:
        """Calculate eyebrow height for expression detection."""
        if len(landmarks) < 16:  # Need eyebrow landmarks
            return 0.0
        
        # Use last landmarks for eyebrows
        eyebrow_points = landmarks[-4:]
        
        if len(eyebrow_points) < 2:
            return 0.0
        
        # Average eyebrow height
        avg_height = np.mean([p[1] for p in eyebrow_points])
        
        # Normalize relative to face (assuming y=0 is top)
        return avg_height
    
    def _estimate_head_orientation(self, landmarks: np.ndarray, face_size: Tuple[int, int]) -> Tuple[float, float, float]:
        """Estimate head orientation (yaw, pitch, roll)."""
        if len(landmarks) < 8:
            return (0.0, 0.0, 0.0)
        
        # Use eye positions for yaw estimation
        left_eye = landmarks[1] if len(landmarks) > 1 else landmarks[0]
        right_eye = landmarks[5] if len(landmarks) > 5 else landmarks[-1]
        
        # Calculate yaw based on eye symmetry
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        face_center_x = face_size[0] / 2
        yaw = (eye_center_x - face_center_x) / face_center_x
        
        # Simplified pitch and roll
        pitch = 0.0  # Would need more sophisticated calculation
        roll = 0.0   # Would need more sophisticated calculation
        
        return (yaw, pitch, roll)
    
    def _determine_facial_states(self, ear_left: float, ear_right: float, 
                                mar: float, eyebrow_height: float) -> List[FacialState]:
        """Determine current facial states based on measurements."""
        states = []
        
        # Eye states
        avg_ear = (ear_left + ear_right) / 2
        if avg_ear < self.EAR_THRESHOLD * 0.7:
            states.append(FacialState.EYES_CLOSED)
        elif avg_ear < self.EAR_THRESHOLD:
            states.append(FacialState.EYES_SQUINTING)
        else:
            states.append(FacialState.EYES_OPEN)
        
        # Mouth states
        if mar > self.MAR_THRESHOLD:
            states.append(FacialState.MOUTH_OPEN)
        elif mar > self.MAR_THRESHOLD * 0.3:
            states.append(FacialState.MOUTH_SMILING)
        else:
            states.append(FacialState.MOUTH_CLOSED)
        
        # Eyebrow states (simplified)
        if eyebrow_height > 0:  # Would need baseline comparison
            states.append(FacialState.EYEBROWS_RAISED)
        
        return states
    
    def _detect_eye_changes(self, prev: FacialFeatures, curr: FacialFeatures) -> Optional[FacialChange]:
        """Detect eye state changes."""
        prev_ear = (prev.eye_aspect_ratio_left + prev.eye_aspect_ratio_right) / 2
        curr_ear = (curr.eye_aspect_ratio_left + curr.eye_aspect_ratio_right) / 2
        
        ear_diff = abs(prev_ear - curr_ear)
        
        if ear_diff > 0.05:  # Significant eye change
            # Determine change type
            if prev_ear > self.EAR_THRESHOLD and curr_ear < self.EAR_THRESHOLD:
                change_type = "eyes_closing"
            elif prev_ear < self.EAR_THRESHOLD and curr_ear > self.EAR_THRESHOLD:
                change_type = "eyes_opening"
            else:
                change_type = "eye_movement"
            
            return FacialChange(
                change_type=change_type,
                severity=min(1.0, ear_diff / 0.2),
                description=f"Eye aspect ratio changed by {ear_diff:.3f}",
                confidence=0.8,
                previous_state=[s for s in prev.facial_states if "EYES" in s.value.upper()],
                current_state=[s for s in curr.facial_states if "EYES" in s.value.upper()]
            )
        
        return None
    
    def _detect_mouth_changes(self, prev: FacialFeatures, curr: FacialFeatures) -> Optional[FacialChange]:
        """Detect mouth state changes."""
        mar_diff = abs(prev.mouth_aspect_ratio - curr.mouth_aspect_ratio)
        
        if mar_diff > 0.1:  # Significant mouth change
            change_type = "mouth_movement"
            
            if prev.mouth_aspect_ratio < self.MAR_THRESHOLD and curr.mouth_aspect_ratio > self.MAR_THRESHOLD:
                change_type = "mouth_opening"
            elif prev.mouth_aspect_ratio > self.MAR_THRESHOLD and curr.mouth_aspect_ratio < self.MAR_THRESHOLD:
                change_type = "mouth_closing"
            
            return FacialChange(
                change_type=change_type,
                severity=min(1.0, mar_diff / 0.5),
                description=f"Mouth aspect ratio changed by {mar_diff:.3f}",
                confidence=0.7,
                previous_state=[s for s in prev.facial_states if "MOUTH" in s.value.upper()],
                current_state=[s for s in curr.facial_states if "MOUTH" in s.value.upper()]
            )
        
        return None
    
    def _detect_eyebrow_changes(self, prev: FacialFeatures, curr: FacialFeatures) -> Optional[FacialChange]:
        """Detect eyebrow position changes."""
        eyebrow_diff = abs(prev.eyebrow_height - curr.eyebrow_height)
        
        if eyebrow_diff > 5:  # Significant eyebrow movement
            return FacialChange(
                change_type="eyebrow_movement",
                severity=min(1.0, eyebrow_diff / 20),
                description=f"Eyebrow position changed by {eyebrow_diff:.1f} pixels",
                confidence=0.6,
                previous_state=[s for s in prev.facial_states if "EYEBROW" in s.value.upper()],
                current_state=[s for s in curr.facial_states if "EYEBROW" in s.value.upper()]
            )
        
        return None
    
    def _detect_orientation_changes(self, prev: FacialFeatures, curr: FacialFeatures) -> Optional[FacialChange]:
        """Detect head orientation changes."""
        yaw_diff = abs(prev.face_orientation[0] - curr.face_orientation[0])
        
        if yaw_diff > 0.1:  # Significant head turn
            return FacialChange(
                change_type="head_orientation",
                severity=min(1.0, yaw_diff / 0.5),
                description=f"Head orientation changed by {yaw_diff:.3f}",
                confidence=0.7,
                previous_state=[],
                current_state=[]
            )
        
        return None
    
    @staticmethod
    def _euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)