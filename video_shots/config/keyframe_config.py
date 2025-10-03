"""Configuration for automatic keyframe selection."""

# Quality assessment weights
QUALITY_WEIGHTS = {
    "visual_quality": 0.25,
    "content_score": 0.30,
    "uniqueness": 0.20,
    "motion_score": 0.15,
    "composition": 0.10
}

# Visual quality thresholds
QUALITY_THRESHOLDS = {
    "sharpness_multiplier": 100,
    "contrast_divisor": 25.5,
    "noise_divisor": 2.55,
    "edge_density_multiplier": 50,
    "color_diversity_multiplier": 20
}

# Scene detection settings
SCENE_DETECTION = {
    "histogram_threshold": 0.3,
    "edge_threshold": 0.4,
    "optical_flow_threshold": 0.5,
    "min_scene_duration": 2.0,  # seconds
    "sample_rate": 0.5  # seconds between samples
}

# Content analysis settings
CONTENT_ANALYSIS = {
    "face_bonus_multiplier": 3,
    "edge_canny_low": 50,
    "edge_canny_high": 150,
    "histogram_bins": [50, 60, 60],
    "histogram_ranges": [0, 180, 0, 256, 0, 256]
}

# Motion detection settings
MOTION_DETECTION = {
    "optical_flow_points": 100,
    "motion_magnitude_divisor": 5,
    "bilateral_filter_d": 9,
    "bilateral_filter_sigma_color": 75,
    "bilateral_filter_sigma_space": 75
}

# Composition assessment
COMPOSITION_RULES = {
    "rule_of_thirds_tolerance": 0.1,  # 10% of frame width/height
    "interest_points_max": 100,
    "good_features_quality": 0.01,
    "good_features_min_distance": 10
}

# Selection strategies
SELECTION_STRATEGIES = {
    "diversity_temporal_weight": 0.6,
    "diversity_visual_weight": 0.4,
    "min_frame_interval": 0.2,  # seconds
    "clustering_oversample_factor": 2
}

# Score interpretation
SCORE_RANGES = {
    "excellent": (8, 10),
    "good": (6, 8),
    "fair": (4, 6),
    "poor": (0, 4)
}

# Frame type preferences
FRAME_TYPE_WEIGHTS = {
    "close_up": 1.2,
    "medium": 1.0,
    "wide": 0.8,
    "face_detected": 1.3,
    "motion_high": 1.1,
    "motion_low": 0.9
}

# Export settings
EXPORT_SETTINGS = {
    "report_precision": 3,
    "timestamp_format": "seconds",
    "include_reasoning": True,
    "include_technical_details": True
}

# Performance settings
PERFORMANCE = {
    "max_frames_in_memory": 50,
    "histogram_cache_size": 100,
    "parallel_processing": False,  # Enable if needed
    "memory_limit_mb": 512
}