#!/usr/bin/env python3
"""
Test script to verify the cleaned up codebase works correctly.
"""

import numpy as np
import tempfile
import os
import sys

def test_audio_utils():
    """Test the new centralized audio utilities."""
    print("Testing audio_utils...")
    
    try:
        from audio_utils import (
            to_float32, sanitize_audio, ensure_stereo, to_mono,
            db_to_linear, linear_to_db, true_peak_db, normalize_peak,
            validate_audio, rms_db, crest_factor_db
        )
        
        # Test audio conversion
        test_int16 = np.array([16384, -16384], dtype=np.int16)
        float_result = to_float32(test_int16)
        assert float_result.dtype == np.float32
        assert np.allclose(float_result, [0.5, -0.5], atol=1e-3)
        
        # Test stereo/mono conversion
        mono = np.array([1, 2, 3], dtype=np.float32)
        stereo = ensure_stereo(mono)
        assert stereo.shape == (3, 2)
        
        back_to_mono = to_mono(stereo)
        assert np.allclose(back_to_mono, mono)
        
        # Test dB conversion
        assert np.isclose(db_to_linear(-6.0), 0.5, atol=1e-3)
        assert np.isclose(linear_to_db(0.5), -6.02, atol=0.1)
        
        # Test validation
        validate_audio(stereo, "test")
        
        print("✅ audio_utils tests passed")
        return True
        
    except Exception as e:
        print(f"❌ audio_utils test failed: {e}")
        return False


def test_configuration():
    """Test the new configuration system."""
    print("Testing configuration system...")
    
    try:
        from config import CONFIG, GlobalConfig
        
        # Test config structure
        assert hasattr(CONFIG, 'audio')
        assert hasattr(CONFIG, 'processing')
        assert hasattr(CONFIG, 'mastering')
        assert hasattr(CONFIG, 'analysis')
        
        # Test some key values
        assert CONFIG.audio.default_bit_depth == "PCM_24"
        assert CONFIG.audio.render_peak_target_dbfs == -1.0
        assert CONFIG.processing.limiter_attack_ms == 1.0
        
        # Test serialization
        config_dict = CONFIG.to_dict()
        assert isinstance(config_dict, dict)
        assert 'audio' in config_dict
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_mastering_orchestrator():
    """Test the cleaned up mastering orchestrator."""
    print("Testing mastering orchestrator...")
    
    try:
        from mastering_orchestrator import (
            LocalMasterProvider, MasterRequest, MasteringOrchestrator
        )
        from config import CONFIG
        
        # Test provider instantiation
        provider = LocalMasterProvider(bit_depth=CONFIG.audio.default_bit_depth)
        assert provider.name == "local"
        assert provider.bit_depth == CONFIG.audio.default_bit_depth
        
        # Test request creation
        request = MasterRequest("/fake/path.wav", style="neutral", strength=0.5)
        assert request.style == "neutral"
        assert request.strength == 0.5
        
        print("✅ Mastering orchestrator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Mastering orchestrator test failed: {e}")
        return False


def test_analysis_module():
    """Test the cleaned up analysis module."""
    print("Testing analysis module...")
    
    try:
        from analysis import analyze_audio_array, health_metrics, stereo_metrics
        from audio_utils import sanitize_audio
        
        # Create test audio
        duration = 1.0  # 1 second
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Stereo sine wave
        left = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
        right = 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz sine
        test_audio = np.column_stack([left, right]).astype(np.float32)
        
        # Test analysis
        report = analyze_audio_array(test_audio, sample_rate)
        assert hasattr(report, 'sr')
        assert hasattr(report, 'basic')
        assert hasattr(report, 'stereo')
        assert report.sr == sample_rate
        
        # Test health metrics
        health = health_metrics(test_audio, sample_rate)
        assert 'peak_dbfs' in health
        assert 'rms_dbfs' in health
        assert 'crest_db' in health
        
        # Test stereo metrics
        stereo_info = stereo_metrics(test_audio)
        assert 'phase_correlation' in stereo_info
        assert 'stereo_width' in stereo_info
        
        print("✅ Analysis module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Analysis module test failed: {e}")
        return False


def test_utils_module():
    """Test the updated utils module."""
    print("Testing utils module...")
    
    try:
        from utils import save_wav_24bit, safe_true_peak, TPGuardResult
        import tempfile
        import soundfile as sf
        
        # Create test audio
        test_audio = np.array([[0.5, -0.5], [0.3, -0.3]], dtype=np.float32)
        sample_rate = 48000
        
        # Test save function
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            saved_path = save_wav_24bit(tmp_path, test_audio, sample_rate)
            assert os.path.exists(saved_path)
            
            # Verify file was saved correctly
            loaded_audio, loaded_sr = sf.read(saved_path)
            assert loaded_sr == sample_rate
            assert loaded_audio.shape == test_audio.shape
            
            # Test true peak guard
            tp_result = safe_true_peak(test_audio * 2.0, sample_rate, ceiling_db=-6.0)
            assert isinstance(tp_result, TPGuardResult)
            assert tp_result.gain_db < 0  # Should apply negative gain
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        print("✅ Utils module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Utils module test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing cleaned up codebase...")
    print("=" * 50)
    
    tests = [
        test_audio_utils,
        test_configuration,
        test_mastering_orchestrator,
        test_analysis_module,
        test_utils_module,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The cleanup was successful.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())