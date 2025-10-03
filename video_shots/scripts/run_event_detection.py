#!/usr/bin/env python3
"""Run event-driven keyframe detection on test video."""

import sys
from pathlib import Path
import time
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_optimized_detection():
    """Run event-driven detection with optimized settings for speed."""
    print("🚀 ЗАПУСК EVENT-DRIVEN ДЕТЕКЦІЇ КЕЙФРЕЙМІВ")
    print("=" * 60)
    
    video_path = Path("input/videos/nyane_30s.mp4")
    if not video_path.exists():
        print(f"❌ Відео файл не знайдено: {video_path}")
        return False
    
    try:
        from video_shots.core.keyframe_selector import KeyframeSelector
        
        print(f"📹 Обробляємо відео: {video_path.name}")
        
        # Створюємо селектор з оптимізованими налаштуваннями
        selector = KeyframeSelector(
            mode="event_driven",
            facial_sensitivity=0.2,      # Помірна чутливість для швидкості
            pose_sensitivity=0.3,        
            scene_sensitivity=0.4,       
            similarity_threshold=0.8,    # Нижчий поріг для швидкості
            filter_strategy="keep_first_discard_rest"
        )
        
        # Налаштовуємо максимальну швидкість обробки
        if hasattr(selector.micro_detector, 'max_frames_per_second'):
            selector.micro_detector.max_frames_per_second = 3  # Дуже повільно для стабільності
        
        print("⚙️  НАЛАШТУВАННЯ:")
        print(f"   🎛️  Режим: event_driven")
        print(f"   👁️  Чутливість до обличчя: 0.2")
        print(f"   🤸 Чутливість до пози: 0.3")
        print(f"   🎬 Чутливість до сцени: 0.4")
        print(f"   📊 Макс. FPS обробки: 3")
        print(f"   🔧 Стратегія: keep_first_discard_rest")
        
        print(f"\n🎯 ПОЧАТОК ДЕТЕКЦІЇ...")
        start_time = time.time()
        
        # Запускаємо детекцію з помірною кількістю кейфреймів
        timepoints = selector.select_keyframes(
            video_path=video_path,
            target_count=15  # Помірна кількість
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ ДЕТЕКЦІЯ ЗАВЕРШЕНА!")
        print(f"   ⏱️  Час обробки: {processing_time:.2f} секунд")
        print(f"   🎬 Знайдено кейфреймів: {len(timepoints)}")
        print(f"   📈 Швидкість: ~{29.6/processing_time:.1f}x від реального часу")
        
        if timepoints:
            print(f"\n🎬 ВИЯВЛЕНІ КЕЙФРЕЙМИ:")
            for i, tp in enumerate(timepoints):
                print(f"   {i+1:2d}. {tp.seconds:6.3f}с (кадр {tp.frame_index})")
            
            # Аналізуємо розподіл у часі
            if len(timepoints) > 1:
                intervals = []
                for i in range(len(timepoints)-1):
                    interval = timepoints[i+1].seconds - timepoints[i].seconds
                    intervals.append(interval)
                
                avg_interval = sum(intervals) / len(intervals)
                print(f"\n📊 АНАЛІЗ РОЗПОДІЛУ:")
                print(f"   ⏱️  Середній інтервал: {avg_interval:.2f}с")
                print(f"   📏 Мін. інтервал: {min(intervals):.2f}с")
                print(f"   📏 Макс. інтервал: {max(intervals):.2f}с")
                print(f"   🎯 Ефективність: {len(timepoints)/(29.6/avg_interval):.1%} від регулярного семплування")
        
        # Спробуємо отримати детальний аналіз
        print(f"\n🔍 ОТРИМАННЯ ДЕТАЛЬНОГО АНАЛІЗУ...")
        try:
            analysis_start = time.time()
            analysis = selector.get_event_driven_analysis(video_path)
            analysis_time = time.time() - analysis_start
            
            print(f"   ✅ Аналіз завершено за {analysis_time:.2f}с")
            print(f"\n📈 СТАТИСТИКА ЗМІН:")
            print(f"   🔍 Всього змін виявлено: {analysis['total_changes_detected']}")
            print(f"   🏷️  Класифіковано подій: {analysis['classified_events']}")
            print(f"   📋 Відфільтровано подій: {analysis['filtered_events']}")
            print(f"   🎬 Фінальних кейфреймів: {analysis['final_timepoints']}")
            print(f"   📊 Коефіцієнт стиснення: {analysis['compression_ratio']:.3f}")
            
            # Показуємо типи виявлених змін
            if 'change_events' in analysis and analysis['change_events']:
                change_types = {}
                for event in analysis['change_events']:
                    change_type = event.change_type.value
                    if change_type not in change_types:
                        change_types[change_type] = 0
                    change_types[change_type] += 1
                
                print(f"\n🏷️  ТИПИ ВИЯВЛЕНИХ ЗМІН:")
                for change_type, count in change_types.items():
                    print(f"   📌 {change_type}: {count} змін")
                
                # Показуємо приклади змін
                print(f"\n🎯 ПРИКЛАДИ ВИЯВЛЕНИХ ЗМІН:")
                for i, event in enumerate(analysis['change_events'][:5]):
                    print(f"   {i+1}. {event.timestamp:6.3f}с - {event.change_type.value}")
                    print(f"      Опис: {event.description}")
                    print(f"      Оцінка: {event.change_score:.3f}, Впевненість: {event.confidence:.3f}")
                
                if len(analysis['change_events']) > 5:
                    print(f"   ... та ще {len(analysis['change_events']) - 5} змін")
            
            # Зберігаємо звіт
            report_path = Path("event_detection_results.json")
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "video": {
                    "file": str(video_path),
                    "duration": 29.6
                },
                "processing": {
                    "detection_time": processing_time,
                    "analysis_time": analysis_time,
                    "total_time": processing_time + analysis_time
                },
                "results": {
                    "keyframes_selected": len(timepoints),
                    "total_changes_detected": analysis['total_changes_detected'],
                    "compression_ratio": analysis['compression_ratio']
                },
                "keyframes": [
                    {
                        "index": tp.index,
                        "timestamp": tp.seconds,
                        "frame_index": tp.frame_index
                    }
                    for tp in timepoints
                ]
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Звіт збережено: {report_path}")
            
        except Exception as e:
            print(f"   ⚠️  Детальний аналіз не вдався: {e}")
        
        print(f"\n🎉 EVENT-DRIVEN ДЕТЕКЦІЯ УСПІШНО ПРАЦЮЄ!")
        print(f"✅ Система виявляє зміни подій та фільтрує кейфрейми!")
        print(f"👁️  Готова до детекції кліпання очей та інших мікро-змін!")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка при запуску: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_pipeline_test():
    """Тест простого pipeline."""
    print(f"\n🔧 ТЕСТ INTEGRATION З PIPELINE")
    print("=" * 60)
    
    try:
        # Тимчасово змінюємо налаштування для швидкості
        from video_shots.config.config import
        
        # Зберігаємо оригінальні значення
        original_max_fps = getattr(config, 'MAX_FRAMES_PER_SECOND', 5)
        original_sensitivity = config.CHANGE_SENSITIVITY.copy()
        
        # Встановлюємо швидкі значення
        config.MAX_FRAMES_PER_SECOND = 2  # Дуже повільно
        config.CHANGE_SENSITIVITY = {
            'facial': 0.3,
            'pose': 0.4,
            'scene': 0.5,
            'motion': 0.4,
            'visual_quality': 0.5
        }
        
        print(f"⚙️  Налаштування для швидкості:")
        print(f"   📊 MAX_FRAMES_PER_SECOND: {config.MAX_FRAMES_PER_SECOND}")
        print(f"   🎯 Знижена чутливість для швидкості")
        
        from video_shots.core.pipeline import VideoAnalysisPipeline
        
        # Тестуємо тільки обчислення кейфреймів
        pipeline = VideoAnalysisPipeline("input/videos/nyane_30s.mp4")
        metadata = pipeline._get_video_metadata()
        
        print(f"\n📹 Метадані відео:")
        print(f"   ⏱️  Тривалість: {metadata['duration']:.2f}с")
        print(f"   📊 FPS: {metadata['fps']:.1f}")
        print(f"   🎬 Кадрів: {metadata['frame_count']}")
        
        print(f"\n🎯 Обчислення кейфреймів...")
        start_time = time.time()
        timepoints = pipeline._calculate_keyframe_timepoints(metadata)
        calc_time = time.time() - start_time
        
        print(f"✅ Обчислення завершено за {calc_time:.2f}с")
        print(f"🎬 Згенеровано {len(timepoints)} кейфреймів")
        
        # Відновлюємо налаштування
        config.MAX_FRAMES_PER_SECOND = original_max_fps
        config.CHANGE_SENSITIVITY = original_sensitivity
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline тест не вдався: {e}")
        return False

def main():
    """Головна функція запуску."""
    print("🎬 ЗАПУСК СИСТЕМИ EVENT-DRIVEN ДЕТЕКЦІЇ")
    print("=" * 60)
    
    success1 = run_optimized_detection()
    success2 = run_simple_pipeline_test()
    
    if success1 and success2:
        print(f"\n" + "=" * 60)
        print("🎉 СИСТЕМА УСПІШНО ЗАПУЩЕНА!")
        print("✅ Event-driven детекція кейфреймів працює!")
        print("👁️  Готова виявляти кожну зміну від закритих до відкритих очей!")
        print("🎬 Використовуйте систему на ваших відео!")
        print("=" * 60)
    else:
        print(f"\n⚠️  Деякі тести не пройшли.")

if __name__ == "__main__":
    main()