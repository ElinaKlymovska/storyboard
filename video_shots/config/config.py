# Keyframe analysis settings
DEFAULT_ANALYSIS_PROMPT = """You are assisting an editor who prepares the Keyframes table. Analyze the provided video frame and return ONLY valid JSON with the following structure:
{
  "scene_id": "scene_01",
  "shot_id": "shot_01",
  "kf_index": 1,
  "timecode": "HH:MM:SS.mmm",
  "thumb_url": "",
  "action_note": "Brief description of what happens in this exact moment",
  "dialogue_or_vo": "Spoken line or voice-over at this moment, or empty string",
  "transition_in_out": {
    "in": "cut",
    "out": "cut"
  },
  "status": "draft",
  "notes": "Any additional editorial notes (can be empty)"
}

Guidelines:
- scene_id and shot_id must be concise snake_case identifiers (use placeholders if unsure).
- timecode must be formatted exactly as HH:MM:SS.mmm and align with the observed moment.
- transition values must be one of: cut, fade, crossfade, glitch, wipe, other.
- status must be either draft or approved (default to draft if uncertain).
- thumb_url can be left blank; pipeline will fill it later.
- If information is unknown, provide an empty string but keep the key.

Return ONLY the JSON object, without extra commentary or markdown."""

# Video shot analysis settings (~5 second intervals)
SHOT_ANALYSIS_PROMPT = """You analyze a 5-second shot segment to populate the Storyboard table. Return ONLY valid JSON:
{
  "scene_id": "scene_01",
  "shot_id": "shot_01",
  "start_tc": "HH:MM:SS.mmm",
  "end_tc": "HH:MM:SS.mmm",
  "shot_type": "WS",
  "objective_or_beat": "Narrative purpose or beat covered in this shot",
  "action": "Short description of on-screen action",
  "camera": "Camera movement or angle (e.g., static, pan left, dolly in)",
  "audio": "VO/dialogue/SFX/music summary or empty string",
  "transition_out": "cut",
  "status": "draft"
}

Rules:
- Use the enumerations: shot_type in {WS, MS, CU, ECU, POV, OTS, Macro}; transition_out in {cut, fade, crossfade, glitch, wipe, match_cut, other}; status in {draft, approved} (default draft).
- timecodes must use HH:MM:SS.mmm format and be consistent (end_tc >= start_tc).
- Provide concise but informative text; keep audio empty if nothing notable.
- scene_id/shot_id must stay consistent with Keyframes output when known; otherwise invent stable placeholders.

Return ONLY the JSON object with no extra commentary."""

# Replicate API settings (image analysis)
REPLICATE_MODEL = "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"
REPLICATE_MAX_TOKENS = 500
REPLICATE_TEMPERATURE = 0.3

# Video analysis settings
VIDEO_ANALYSIS_MODEL = "lucataco/videollama3-7b:34a1f45f7068f7121a5b47c91f2d7e06c298850767f76f96660450a0a3bd5bbe"
VIDEO_ANALYSIS_INTERVAL = 0.5

# Event-driven keyframe detection settings
MICRO_CHANGE_DETECTION = True
KEYFRAME_SELECTION_MODE = "event_driven"  # "interval_based" or "event_driven"

# Micro-change detection sensitivity settings
CHANGE_SENSITIVITY = {
    'facial': 0.15,        # Very sensitive for facial changes (eye blinks, expressions)
    'pose': 0.25,          # Medium sensitivity for pose changes
    'scene': 0.4,          # Lower sensitivity for major scene changes
    'motion': 0.3,         # Motion change sensitivity
    'visual_quality': 0.35 # Visual quality change sensitivity
}

# Frame similarity and filtering settings
SIMILARITY_THRESHOLD = 0.80     # Frames more similar than this are considered "same event"
MAX_FRAMES_PER_SECOND = 5       # Performance limit for event detection (reduced for speed)
MIN_SEQUENCE_LENGTH = 3         # Minimum frames to form a similar sequence
MAX_SEQUENCE_DURATION = 2.0     # Maximum duration for a single sequence (seconds)

# Intelligent filtering strategy
FILTER_STRATEGY = "keep_first_discard_rest"  # Options: "keep_first_discard_rest", "keep_best_in_sequence", "adaptive_sampling", "quality_based"

# Event classification weights
EVENT_CLASSIFICATION_WEIGHTS = {
    'facial_weight': 0.4,      # Weight for facial change importance
    'motion_weight': 0.3,      # Weight for motion change importance  
    'scene_weight': 0.2,       # Weight for scene change importance
    'temporal_weight': 0.1     # Weight for temporal distribution
}

# Event filtering thresholds
MIN_SIGNIFICANCE_THRESHOLD = 0.3    # Minimum significance score to keep event
MIN_CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence for change detection

# Event priority settings - which priorities to keep
KEEP_EVENT_PRIORITIES = ["critical", "high", "medium"]  # Options: "critical", "high", "medium", "low"

# Event category settings - which categories to prioritize
PRIORITIZE_EVENT_CATEGORIES = [
    "facial_expression",       # Eye blinks, smiles, etc.
    "scene_transition",        # Major scene changes
    "body_movement",           # Pose changes, gestures
    "lighting_change"          # Significant lighting changes
]

# Video processing settings
MAX_ITERATIONS = 10_000_000
MAX_FRAMES_WARNING = 1000
DEFAULT_FPS_FOR_TIMECODE = 25.0

# Image settings
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_JPEG_QUALITY = 95
DEFAULT_FONT_SCALE = 0.8
DEFAULT_FONT_THICKNESS = 2
DEFAULT_PADDING = 12

# Panel and text colors
PANEL_BG_COLOR = (0, 0, 0)
PANEL_ALPHA = 0.6
TEXT_COLOR = (255, 255, 255)

# Time constants
CYRILLIC_S_LOWER = 'с'
CYRILLIC_S_UPPER = 'С'

# Video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

# Default settings
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SCREENSHOT_PREFIX = "shot"
DEFAULT_SCREENSHOT_FORMAT = "png"
DEFAULT_QUALITY = 95
