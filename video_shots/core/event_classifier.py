"""Event classification for categorizing detected changes."""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

from video_shots.core.micro_change_detector import ChangeEvent, ChangeType
from video_shots.core.facial_analyzer import FacialChange


class EventPriority(Enum):
    """Priority levels for different types of events."""
    CRITICAL = "critical"      # Major scene changes, cuts
    HIGH = "high"             # Facial expressions, significant poses
    MEDIUM = "medium"         # Moderate motion, lighting changes
    LOW = "low"               # Minor adjustments, noise


class EventCategory(Enum):
    """Categories of detected events."""
    SCENE_TRANSITION = "scene_transition"
    FACIAL_EXPRESSION = "facial_expression"
    BODY_MOVEMENT = "body_movement"
    CAMERA_MOVEMENT = "camera_movement"
    LIGHTING_CHANGE = "lighting_change"
    OBJECT_INTERACTION = "object_interaction"
    AUDIO_VISUAL_SYNC = "audio_visual_sync"
    TECHNICAL_CHANGE = "technical_change"


@dataclass
class ClassifiedEvent:
    """A change event with classification metadata."""
    original_event: ChangeEvent
    category: EventCategory
    priority: EventPriority
    significance_score: float  # 0.0 to 1.0
    temporal_importance: float  # 0.0 to 1.0
    narrative_relevance: float  # 0.0 to 1.0
    technical_quality: float   # 0.0 to 1.0
    recommended_action: str    # "keep", "discard", "merge", "enhance"
    classification_confidence: float
    tags: List[str]


@dataclass
class EventContext:
    """Contextual information for event classification."""
    surrounding_events: List[ChangeEvent]
    time_since_last_major: float
    video_duration: float
    current_position_ratio: float  # 0.0 to 1.0 (position in video)
    scene_stability: float  # How stable the scene has been
    motion_trend: str  # "increasing", "decreasing", "stable"


class EventClassifier:
    """Classifies detected change events for intelligent filtering and prioritization."""
    
    def __init__(self,
                 facial_weight: float = 0.4,
                 motion_weight: float = 0.3,
                 scene_weight: float = 0.2,
                 temporal_weight: float = 0.1,
                 min_significance_threshold: float = 0.3):
        """
        Initialize event classifier.
        
        Args:
            facial_weight: Weight for facial change importance (0-1)
            motion_weight: Weight for motion change importance (0-1)
            scene_weight: Weight for scene change importance (0-1)
            temporal_weight: Weight for temporal distribution importance (0-1)
            min_significance_threshold: Minimum significance score to keep event
        """
        self.facial_weight = facial_weight
        self.motion_weight = motion_weight
        self.scene_weight = scene_weight
        self.temporal_weight = temporal_weight
        self.min_significance_threshold = min_significance_threshold
        
        # Normalization factors
        total_weight = facial_weight + motion_weight + scene_weight + temporal_weight
        if total_weight > 0:
            self.facial_weight /= total_weight
            self.motion_weight /= total_weight
            self.scene_weight /= total_weight
            self.temporal_weight /= total_weight
    
    def classify_events(self, 
                       events: List[ChangeEvent],
                       video_duration: float) -> List[ClassifiedEvent]:
        """
        Classify a list of change events.
        
        Args:
            events: List of detected change events
            video_duration: Total video duration in seconds
            
        Returns:
            List of classified events
        """
        if not events:
            return []
        
        classified_events = []
        
        for i, event in enumerate(events):
            # Create context for this event
            context = self._create_event_context(events, i, video_duration)
            
            # Classify the event
            classified_event = self._classify_single_event(event, context)
            classified_events.append(classified_event)
        
        # Post-process classifications (adjust based on global context)
        self._post_process_classifications(classified_events)
        
        return classified_events
    
    def _create_event_context(self, 
                             events: List[ChangeEvent], 
                             current_index: int,
                             video_duration: float) -> EventContext:
        """Create contextual information for an event."""
        current_event = events[current_index]
        
        # Get surrounding events
        window_size = 5
        start_idx = max(0, current_index - window_size)
        end_idx = min(len(events), current_index + window_size + 1)
        surrounding_events = events[start_idx:end_idx]
        
        # Calculate time since last major event
        time_since_last_major = 0.0
        for i in range(current_index - 1, -1, -1):
            prev_event = events[i]
            if (prev_event.change_score > 0.7 or 
                prev_event.change_type in [ChangeType.SCENE_COMPOSITION, ChangeType.VISUAL_QUALITY]):
                time_since_last_major = current_event.timestamp - prev_event.timestamp
                break
            if i == 0:
                time_since_last_major = current_event.timestamp
        
        # Calculate position in video
        position_ratio = current_event.timestamp / video_duration if video_duration > 0 else 0.0
        
        # Calculate scene stability (variance in recent change scores)
        recent_scores = [e.change_score for e in surrounding_events]
        scene_stability = 1.0 - (np.std(recent_scores) if len(recent_scores) > 1 else 0.0)
        
        # Determine motion trend
        motion_trend = self._analyze_motion_trend(surrounding_events, current_index)
        
        return EventContext(
            surrounding_events=surrounding_events,
            time_since_last_major=time_since_last_major,
            video_duration=video_duration,
            current_position_ratio=position_ratio,
            scene_stability=scene_stability,
            motion_trend=motion_trend
        )
    
    def _classify_single_event(self, 
                              event: ChangeEvent, 
                              context: EventContext) -> ClassifiedEvent:
        """Classify a single event with its context."""
        
        # Determine category
        category = self._determine_category(event)
        
        # Determine priority
        priority = self._determine_priority(event, category, context)
        
        # Calculate significance score
        significance_score = self._calculate_significance_score(event, context)
        
        # Calculate temporal importance
        temporal_importance = self._calculate_temporal_importance(event, context)
        
        # Calculate narrative relevance
        narrative_relevance = self._calculate_narrative_relevance(event, category, context)
        
        # Calculate technical quality
        technical_quality = self._calculate_technical_quality(event)
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(
            event, significance_score, temporal_importance, narrative_relevance
        )
        
        # Calculate classification confidence
        classification_confidence = self._calculate_classification_confidence(event, context)
        
        # Generate tags
        tags = self._generate_tags(event, category, priority, context)
        
        return ClassifiedEvent(
            original_event=event,
            category=category,
            priority=priority,
            significance_score=significance_score,
            temporal_importance=temporal_importance,
            narrative_relevance=narrative_relevance,
            technical_quality=technical_quality,
            recommended_action=recommended_action,
            classification_confidence=classification_confidence,
            tags=tags
        )
    
    def _determine_category(self, event: ChangeEvent) -> EventCategory:
        """Determine the category of an event."""
        if event.change_type == ChangeType.FACIAL_EXPRESSION:
            return EventCategory.FACIAL_EXPRESSION
        elif event.change_type == ChangeType.POSE_CHANGE:
            return EventCategory.BODY_MOVEMENT
        elif event.change_type == ChangeType.SCENE_COMPOSITION:
            if event.change_score > 0.7:
                return EventCategory.SCENE_TRANSITION
            else:
                return EventCategory.CAMERA_MOVEMENT
        elif event.change_type == ChangeType.VISUAL_QUALITY:
            return EventCategory.LIGHTING_CHANGE
        elif event.change_type == ChangeType.MOTION_CHANGE:
            if event.change_score > 0.6:
                return EventCategory.BODY_MOVEMENT
            else:
                return EventCategory.CAMERA_MOVEMENT
        else:
            return EventCategory.TECHNICAL_CHANGE
    
    def _determine_priority(self, 
                           event: ChangeEvent, 
                           category: EventCategory,
                           context: EventContext) -> EventPriority:
        """Determine the priority of an event."""
        base_priority_score = event.change_score * event.confidence
        
        # Category-based priority adjustments
        if category == EventCategory.SCENE_TRANSITION:
            priority_score = base_priority_score + 0.3
        elif category == EventCategory.FACIAL_EXPRESSION:
            priority_score = base_priority_score + 0.2
        elif category in [EventCategory.BODY_MOVEMENT, EventCategory.CAMERA_MOVEMENT]:
            priority_score = base_priority_score + 0.1
        else:
            priority_score = base_priority_score
        
        # Context-based adjustments
        if context.time_since_last_major > 5.0:  # Long time since last major event
            priority_score += 0.1
        
        if context.current_position_ratio < 0.1 or context.current_position_ratio > 0.9:
            # Beginning or end of video
            priority_score += 0.1
        
        # Determine priority level
        if priority_score >= 0.8:
            return EventPriority.CRITICAL
        elif priority_score >= 0.6:
            return EventPriority.HIGH
        elif priority_score >= 0.4:
            return EventPriority.MEDIUM
        else:
            return EventPriority.LOW
    
    def _calculate_significance_score(self, 
                                    event: ChangeEvent, 
                                    context: EventContext) -> float:
        """Calculate overall significance score for an event."""
        base_score = event.change_score * event.confidence
        
        # Adjust based on change type
        type_multiplier = {
            ChangeType.FACIAL_EXPRESSION: self.facial_weight * 4,
            ChangeType.POSE_CHANGE: self.motion_weight * 3,
            ChangeType.SCENE_COMPOSITION: self.scene_weight * 4,
            ChangeType.VISUAL_QUALITY: 0.7,
            ChangeType.MOTION_CHANGE: self.motion_weight * 2
        }.get(event.change_type, 1.0)
        
        adjusted_score = base_score * type_multiplier
        
        # Context adjustments
        if context.time_since_last_major > 3.0:
            adjusted_score *= 1.2  # Boost if no recent major events
        
        if context.scene_stability > 0.8:
            adjusted_score *= 1.1  # Boost if scene was stable
        
        # Temporal position adjustments
        if 0.1 <= context.current_position_ratio <= 0.9:
            adjusted_score *= 1.0  # Middle of video - no adjustment
        else:
            adjusted_score *= 1.2  # Beginning or end - boost
        
        return max(0.0, min(1.0, adjusted_score))
    
    def _calculate_temporal_importance(self, 
                                     event: ChangeEvent, 
                                     context: EventContext) -> float:
        """Calculate temporal importance of an event."""
        # Base temporal importance
        temporal_score = self.temporal_weight
        
        # Adjust based on time since last major event
        if context.time_since_last_major > 5.0:
            temporal_score += 0.3
        elif context.time_since_last_major < 1.0:
            temporal_score -= 0.2
        
        # Adjust based on position in video
        # More important at beginning, end, and thirds
        position = context.current_position_ratio
        if position < 0.1 or position > 0.9:
            temporal_score += 0.2
        elif abs(position - 1/3) < 0.05 or abs(position - 2/3) < 0.05:
            temporal_score += 0.1
        
        return max(0.0, min(1.0, temporal_score))
    
    def _calculate_narrative_relevance(self, 
                                     event: ChangeEvent,
                                     category: EventCategory,
                                     context: EventContext) -> float:
        """Calculate narrative relevance of an event."""
        base_relevance = 0.5
        
        # Category-based relevance
        if category in [EventCategory.FACIAL_EXPRESSION, EventCategory.SCENE_TRANSITION]:
            base_relevance = 0.8
        elif category in [EventCategory.BODY_MOVEMENT, EventCategory.OBJECT_INTERACTION]:
            base_relevance = 0.6
        elif category == EventCategory.LIGHTING_CHANGE:
            base_relevance = 0.4
        
        # Adjust based on event description
        description = event.description.lower()
        if any(keyword in description for keyword in ["eye", "blink", "smile", "expression"]):
            base_relevance += 0.2
        if any(keyword in description for keyword in ["cut", "transition", "scene"]):
            base_relevance += 0.3
        
        return max(0.0, min(1.0, base_relevance))
    
    def _calculate_technical_quality(self, event: ChangeEvent) -> float:
        """Calculate technical quality score for an event."""
        # Base quality from confidence
        quality_score = event.confidence
        
        # Adjust based on change score consistency
        if 0.3 <= event.change_score <= 0.9:
            quality_score += 0.1  # Good range, not too low or suspiciously high
        
        # Adjust based on change type reliability
        type_reliability = {
            ChangeType.SCENE_COMPOSITION: 0.9,
            ChangeType.VISUAL_QUALITY: 0.8,
            ChangeType.FACIAL_EXPRESSION: 0.7,
            ChangeType.MOTION_CHANGE: 0.6,
            ChangeType.POSE_CHANGE: 0.5
        }.get(event.change_type, 0.5)
        
        quality_score = (quality_score + type_reliability) / 2
        
        return max(0.0, min(1.0, quality_score))
    
    def _determine_recommended_action(self, 
                                    event: ChangeEvent,
                                    significance_score: float,
                                    temporal_importance: float,
                                    narrative_relevance: float) -> str:
        """Determine recommended action for an event."""
        combined_score = (significance_score + temporal_importance + narrative_relevance) / 3
        
        if combined_score >= 0.7:
            return "keep"
        elif combined_score >= 0.5:
            if event.change_type == ChangeType.FACIAL_EXPRESSION:
                return "keep"  # Prefer to keep facial expressions
            else:
                return "merge"  # Consider merging with nearby events
        elif combined_score >= 0.3:
            return "merge"
        else:
            return "discard"
    
    def _calculate_classification_confidence(self, 
                                           event: ChangeEvent, 
                                           context: EventContext) -> float:
        """Calculate confidence in the classification."""
        base_confidence = event.confidence
        
        # Adjust based on context consistency
        if context.scene_stability > 0.7:
            base_confidence += 0.1  # More confident in stable scenes
        
        # Adjust based on surrounding events
        surrounding_scores = [e.change_score for e in context.surrounding_events]
        if len(surrounding_scores) > 1:
            score_consistency = 1.0 - np.std(surrounding_scores)
            base_confidence = (base_confidence + score_consistency) / 2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_tags(self, 
                      event: ChangeEvent,
                      category: EventCategory,
                      priority: EventPriority,
                      context: EventContext) -> List[str]:
        """Generate descriptive tags for an event."""
        tags = []
        
        # Add category and priority tags
        tags.append(category.value)
        tags.append(priority.value)
        
        # Add change type specific tags
        if event.change_type == ChangeType.FACIAL_EXPRESSION:
            if "eye" in event.description.lower():
                tags.append("eye_movement")
            if "mouth" in event.description.lower():
                tags.append("mouth_movement")
            if "blink" in event.description.lower():
                tags.append("blink")
        
        # Add temporal tags
        if context.time_since_last_major > 5.0:
            tags.append("isolated_event")
        elif context.time_since_last_major < 1.0:
            tags.append("clustered_event")
        
        # Add position tags
        if context.current_position_ratio < 0.1:
            tags.append("video_beginning")
        elif context.current_position_ratio > 0.9:
            tags.append("video_end")
        elif 0.4 <= context.current_position_ratio <= 0.6:
            tags.append("video_middle")
        
        # Add quality tags
        if event.change_score > 0.8:
            tags.append("high_magnitude")
        elif event.change_score < 0.3:
            tags.append("low_magnitude")
        
        if event.confidence > 0.8:
            tags.append("high_confidence")
        elif event.confidence < 0.5:
            tags.append("low_confidence")
        
        return tags
    
    def _analyze_motion_trend(self, 
                             surrounding_events: List[ChangeEvent], 
                             current_index: int) -> str:
        """Analyze motion trend in surrounding events."""
        if len(surrounding_events) < 3:
            return "stable"
        
        # Find relative position of current event in surrounding events
        current_global_idx = None
        for i, event in enumerate(surrounding_events):
            if i == current_index or (current_index >= len(surrounding_events) and i == len(surrounding_events) - 1):
                current_global_idx = i
                break
        
        if current_global_idx is None or current_global_idx < 1:
            return "stable"
        
        # Compare recent motion scores
        recent_scores = []
        for event in surrounding_events:
            if event.change_type == ChangeType.MOTION_CHANGE:
                recent_scores.append(event.change_score)
        
        if len(recent_scores) < 2:
            return "stable"
        
        # Calculate trend
        if len(recent_scores) >= 2:
            if recent_scores[-1] > recent_scores[0] * 1.2:
                return "increasing"
            elif recent_scores[-1] < recent_scores[0] * 0.8:
                return "decreasing"
        
        return "stable"
    
    def _post_process_classifications(self, classified_events: List[ClassifiedEvent]) -> None:
        """Post-process classifications to ensure global consistency."""
        if not classified_events:
            return
        
        # Ensure minimum distribution of kept events
        keep_events = [e for e in classified_events if e.recommended_action == "keep"]
        total_events = len(classified_events)
        
        if total_events > 0:
            keep_ratio = len(keep_events) / total_events
            
            # If too few events are being kept, promote some merge/discard events
            if keep_ratio < 0.1:  # Less than 10% kept
                merge_events = [e for e in classified_events if e.recommended_action == "merge"]
                merge_events.sort(key=lambda x: x.significance_score, reverse=True)
                
                # Promote top merge events to keep
                promote_count = min(len(merge_events), max(1, int(total_events * 0.1)))
                for i in range(promote_count):
                    merge_events[i].recommended_action = "keep"
            
            # If too many events are being kept, demote some
            elif keep_ratio > 0.5:  # More than 50% kept
                keep_events.sort(key=lambda x: x.significance_score)
                
                # Demote lowest scoring keep events
                demote_count = int(total_events * 0.3)  # Target 30% kept
                for i in range(min(demote_count, len(keep_events))):
                    keep_events[i].recommended_action = "merge"


def filter_by_classification(classified_events: List[ClassifiedEvent],
                           min_significance: float = 0.3,
                           priority_filter: Optional[List[EventPriority]] = None,
                           category_filter: Optional[List[EventCategory]] = None) -> List[ClassifiedEvent]:
    """
    Filter classified events based on various criteria.
    
    Args:
        classified_events: List of classified events
        min_significance: Minimum significance score to keep
        priority_filter: Keep only events with these priorities
        category_filter: Keep only events in these categories
        
    Returns:
        Filtered list of classified events
    """
    filtered = []
    
    for event in classified_events:
        # Check significance threshold
        if event.significance_score < min_significance:
            continue
        
        # Check priority filter
        if priority_filter and event.priority not in priority_filter:
            continue
        
        # Check category filter
        if category_filter and event.category not in category_filter:
            continue
        
        # Check recommended action
        if event.recommended_action == "discard":
            continue
        
        filtered.append(event)
    
    return filtered


def generate_classification_report(classified_events: List[ClassifiedEvent], 
                                 output_path) -> None:
    """Generate a detailed classification report."""
    import json
    from pathlib import Path
    
    # Calculate statistics
    total_events = len(classified_events)
    
    if total_events == 0:
        return
    
    category_counts = {}
    priority_counts = {}
    action_counts = {}
    
    for event in classified_events:
        category_counts[event.category.value] = category_counts.get(event.category.value, 0) + 1
        priority_counts[event.priority.value] = priority_counts.get(event.priority.value, 0) + 1
        action_counts[event.recommended_action] = action_counts.get(event.recommended_action, 0) + 1
    
    avg_significance = sum(e.significance_score for e in classified_events) / total_events
    avg_confidence = sum(e.classification_confidence for e in classified_events) / total_events
    
    report = {
        "classification_summary": {
            "total_events": total_events,
            "average_significance": round(avg_significance, 3),
            "average_confidence": round(avg_confidence, 3),
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "action_distribution": action_counts
        },
        "classified_events": [
            {
                "timestamp": f"{event.original_event.timestamp:.3f}s",
                "frame_index": event.original_event.frame_index,
                "category": event.category.value,
                "priority": event.priority.value,
                "significance_score": round(event.significance_score, 3),
                "temporal_importance": round(event.temporal_importance, 3),
                "narrative_relevance": round(event.narrative_relevance, 3),
                "technical_quality": round(event.technical_quality, 3),
                "recommended_action": event.recommended_action,
                "confidence": round(event.classification_confidence, 3),
                "description": event.original_event.description,
                "tags": event.tags
            }
            for event in classified_events
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)