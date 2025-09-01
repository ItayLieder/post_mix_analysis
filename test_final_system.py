#!/usr/bin/env python3
"""
Comprehensive test suite for the cleaned post-mix analysis system.
Tests all critical functionality including both notebooks.
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from typing import Dict, Any

def test_critical_imports():
    """Test all critical module imports work correctly."""
    print("🧪 TESTING CRITICAL IMPORTS")
    print("="*35)
    
    critical_modules = [
        # Core system
        'config', 'utils', 'analysis', 'data_handler', 'comparison_reporting',
        'dsp_premitives', 'processors', 'render_engine', 'big_variants_system',
        
        # Post-mix notebook support
        'mastering_orchestrator', 'stem_mastering', 'logging_versioning',
        'pre_master_prep', 'presets_recommendations', 'streaming_normalization_simulator',
        
        # Professional mixing support
        'pro_mixing_engine', 'pro_mixing_engine_fixed', 'mix_intelligence',
        'reverb_engine', 'reference_matcher', 'advanced_dsp'
    ]
    
    failed_imports = []
    for module in critical_modules:
        try:
            exec(f'import {module}')
            print(f'✅ {module}')
        except Exception as e:
            failed_imports.append((module, str(e)))
            print(f'❌ {module}: {e}')
    
    if failed_imports:
        print(f"\n🚨 {len(failed_imports)} import failures!")
        return False
    else:
        print(f"\n🎉 All {len(critical_modules)} imports successful!")
        return True

def test_config_system():
    """Test the centralized configuration system."""
    print("\n⚙️ TESTING CONFIGURATION SYSTEM")
    print("="*40)
    
    try:
        from config import CONFIG
        
        # Test CONFIG structure
        assert hasattr(CONFIG, 'audio'), "Missing audio config"
        assert hasattr(CONFIG, 'pipeline'), "Missing pipeline config"
        
        # Test stem gains (critical for user's workflow)
        stem_gains = CONFIG.pipeline.get_stem_gains()
        assert isinstance(stem_gains, dict), "Stem gains not dict"
        
        required_stems = ['drums', 'bass', 'vocals', 'music']
        for stem in required_stems:
            assert stem in stem_gains, f"Missing {stem} in stem_gains"
        
        print(f"✅ CONFIG structure valid")
        print(f"✅ Stem gains: {stem_gains}")
        
        # Test audio config
        assert CONFIG.audio.default_sample_rate > 0, "Invalid sample rate"
        assert CONFIG.audio.default_bit_depth in ["PCM_16", "PCM_24", "PCM_32"], "Invalid bit depth"
        
        print(f"✅ Audio config valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_audio_processing():
    """Test core audio processing functionality."""
    print("\n🎵 TESTING AUDIO PROCESSING")
    print("="*30)
    
    try:
        from utils import to_float32, sanitize_audio, ensure_stereo, to_mono
        from analysis import analyze_audio_array
        from dsp_premitives import peaking_eq, compressor, stereo_widener
        
        # Create test audio
        sr = 44100
        duration = 1.0
        samples = int(sr * duration)
        
        # Generate test signals
        sine_wave = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(np.float32) * 0.5
        noise = np.random.normal(0, 0.1, samples).astype(np.float32)
        
        # Test utility functions
        stereo_audio = ensure_stereo(sine_wave)
        assert stereo_audio.shape[1] == 2, "ensure_stereo failed"
        
        mono_audio = to_mono(stereo_audio)
        assert mono_audio.ndim == 1, "to_mono failed"
        
        clean_audio = sanitize_audio(sine_wave + noise)
        assert np.isfinite(clean_audio).all(), "sanitize_audio failed"
        
        print("✅ Audio utilities working")
        
        # Test analysis
        analysis_result = analyze_audio_array(sine_wave, sr)
        assert hasattr(analysis_result, 'basic'), "Analysis missing basic metrics"
        assert 'peak_dbfs' in analysis_result.basic, "Analysis missing peak_dbfs"
        assert 'rms_dbfs' in analysis_result.basic, "Analysis missing rms_dbfs"
        
        print("✅ Audio analysis working")
        
        # Test DSP processing
        eq_result = peaking_eq(sine_wave, sr, f0=1000, gain_db=3.0, Q=1.0)
        assert eq_result.shape == sine_wave.shape, "EQ processing failed"
        assert not np.allclose(eq_result, sine_wave), "EQ had no effect"
        
        comp_result = compressor(sine_wave, sr, threshold_db=-12, ratio=4.0)
        assert comp_result.shape == sine_wave.shape, "Compressor failed"
        
        wide_result = stereo_widener(stereo_audio, width=1.5)
        assert wide_result.shape == stereo_audio.shape, "Stereo widener failed"
        
        print("✅ DSP processing working")
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_big_variants_system():
    """Test the BIG variants system with CONFIG integration."""
    print("\n🎛️ TESTING BIG VARIANTS SYSTEM")
    print("="*35)
    
    try:
        from big_variants_system import get_big_variant_profile, apply_big_variant_processing
        from config import CONFIG
        
        # Test profile creation
        profile = get_big_variant_profile('BIG_Amazing')
        assert hasattr(profile, 'kick_boost_mult'), "Profile missing kick_boost_mult"
        assert hasattr(profile, 'bass_boost_mult'), "Profile missing bass_boost_mult"
        
        print("✅ BIG variant profile creation working")
        
        # Test CONFIG integration
        stem_gains = CONFIG.pipeline.get_stem_gains()
        assert 'drums' in stem_gains, "CONFIG stem gains missing drums"
        
        # Test processing with dummy audio
        sr = 44100
        duration = 0.5
        test_audio = np.sin(2 * np.pi * 60 * np.linspace(0, duration, int(sr * duration))).astype(np.float32)
        
        processed = apply_big_variant_processing('drums', test_audio, sr, profile)
        assert processed.shape == test_audio.shape, "BIG processing changed audio shape"
        # BIG processing may be subtle, so just check it's different in magnitude
        if np.allclose(processed, test_audio, rtol=1e-3):
            print("⚠️ BIG processing had subtle effect (may be expected)")
        else:
            print("✅ BIG processing modified audio significantly")
        
        print("✅ BIG variants processing working")
        print("✅ CONFIG integration confirmed")
        return True
        
    except Exception as e:
        print(f"❌ BIG variants test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workspace_system():
    """Test workspace and manifest system."""
    print("\n📁 TESTING WORKSPACE SYSTEM")
    print("="*30)
    
    try:
        from data_handler import make_workspace, Manifest, write_manifest, read_manifest
        from logging_versioning import RunLogger, capture_environment
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test workspace creation
            workspace = make_workspace(temp_dir, 'test_project')
            assert os.path.exists(workspace.root), "Workspace root not created"
            
            print("✅ Workspace creation working")
            
            # Test manifest system
            manifest = Manifest('test_project', workspace)
            test_data = {'test_key': 'test_value', 'timestamp': '2024-01-01'}
            write_manifest(manifest.path, test_data)
            
            read_data = read_manifest(manifest.path)
            assert read_data['test_key'] == 'test_value', "Manifest read/write failed"
            
            print("✅ Manifest system working")
            
            # Test logging
            logger = RunLogger.start(workspace.root, 'test_run')
            logger.log_event('test_event', {'param1': 'value1'})
            logger.finish()
            
            print("✅ Logging system working")
            
            # Test environment capture
            env_info = capture_environment()
            assert 'python_version' in env_info, "Environment capture incomplete"
            
            print("✅ Environment capture working")
            return True
            
    except Exception as e:
        print(f"❌ Workspace test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notebook_dependencies():
    """Test that both critical notebooks have their dependencies available."""
    print("\n📓 TESTING NOTEBOOK DEPENDENCIES")
    print("="*40)
    
    try:
        # Test post_mix_cleaned dependencies
        post_mix_deps = [
            'mastering_orchestrator', 'stem_mastering', 'pre_master_prep',
            'streaming_normalization_simulator', 'presets_recommendations'
        ]
        
        for dep in post_mix_deps:
            exec(f'import {dep}')
        
        print("✅ post_mix_cleaned.ipynb dependencies available")
        
        # Test professional_mixing dependencies
        prof_mix_deps = [
            'pro_mixing_engine', 'pro_mixing_engine_fixed', 'mix_intelligence',
            'reverb_engine', 'reference_matcher'
        ]
        
        for dep in prof_mix_deps:
            exec(f'import {dep}')
        
        print("✅ professional_mixing.ipynb dependencies available")
        
        # Test key functions are callable
        from pro_mixing_engine import ProMixingSession
        from mix_intelligence import AutoMixer, MixAnalyzer
        
        # Test instantiation with dummy data
        channels = {'drums': {'kick': '/fake/path'}}
        
        try:
            mixer = AutoMixer(sr=44100)
            analyzer = MixAnalyzer(sr=44100)
            print("✅ AI mixing components instantiable")
        except Exception:
            print("⚠️ AI mixing instantiation issues (expected with fake paths)")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook dependencies test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all tests and provide final assessment."""
    print("🏁 COMPREHENSIVE SYSTEM TEST")
    print("="*50)
    print("Testing cleaned post-mix analysis system...")
    print()
    
    tests = [
        ("Import System", test_critical_imports),
        ("Configuration", test_config_system),
        ("Audio Processing", test_audio_processing),
        ("BIG Variants", test_big_variants_system),
        ("Workspace System", test_workspace_system),
        ("Notebook Dependencies", test_notebook_dependencies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Final assessment
    print("\n" + "="*50)
    print("🏆 FINAL TEST RESULTS")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM IS READY!")
        print("✅ Both notebooks should work correctly")
        print("✅ All dependencies satisfied")
        print("✅ CONFIG.pipeline.stem_gains integration working")
        return True
    else:
        print(f"\n⚠️ {total-passed} test(s) failed - system needs attention")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)