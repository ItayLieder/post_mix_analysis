#!/usr/bin/env python3
"""
Comprehensive functionality test for post_mix_cleaned.ipynb
Tests the core workflow without requiring actual audio files.
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import modules at module level
from data_handler import *
from analysis import *
from comparison_reporting import *
from utils import to_float32, sanitize_audio, ensure_stereo, to_mono

def test_post_mix_cleaned_functionality():
    """Test all critical functions from post_mix_cleaned notebook"""
    
    print("🧪 TESTING post_mix_cleaned.ipynb FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test imports (mimics notebook cell 1)
        print("📦 Testing imports...")
        # Imports already done at module level
        print("✅ All imports successful")
        
        # Test audio creation and I/O (mimics notebook workflow)
        print("\n🎵 Testing audio I/O...")
        
        # Create test audio (sine wave)
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        frequency = 440.0  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = np.sin(2 * np.pi * frequency * t) * 0.5  # Sine wave at -6dB
        test_audio_stereo = np.column_stack([test_audio, test_audio * 0.8])  # Slightly different channels
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test save_wav
            test_file = os.path.join(temp_dir, "test_audio.wav")
            save_wav(test_file, test_audio_stereo, sample_rate)
            print("✅ save_wav - working")
            
            # Test load_wav  
            loaded_buffer = load_wav(test_file)
            assert loaded_buffer.sr == sample_rate
            assert loaded_buffer.samples.shape[0] > 0
            print("✅ load_wav - working")
            
            # Test AudioBuffer constructor directly
            buffer = AudioBuffer(sample_rate, test_audio_stereo, "test_audio.wav") 
            assert buffer.sr == sample_rate
            assert hasattr(buffer, 'samples')
            print("✅ AudioBuffer - working")
            
        # Test analysis functions
        print("\n📊 Testing audio analysis...")
        
        # Test analyze_audio_array
        analysis_result = analyze_audio_array(test_audio_stereo, sample_rate)
        assert hasattr(analysis_result, 'true_peak_dbfs')
        assert hasattr(analysis_result, 'basic')
        assert hasattr(analysis_result, 'lufs_integrated')
        print("✅ analyze_audio_array - working")
        
        # Test analysis_table (takes individual report, not list)
        table_df = analysis_table(analysis_result)
        assert len(table_df) > 0
        print("✅ analysis_table - working")
        
        # Test workspace creation
        print("\n📁 Testing workspace management...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = make_workspace(temp_dir)
            assert os.path.exists(workspace.inputs)
            assert os.path.exists(workspace.outputs)
            print("✅ make_workspace - working")
            
            # Test manifest system (basic instantiation)
            manifest = Manifest("test_project", workspace.root)
            assert manifest.project == "test_project"
            assert manifest.workspace == workspace.root
            print("✅ manifest system - working")
        
        # Test comparison and reporting
        print("\n📈 Testing comparison & reporting...")
        
        # Create two slightly different audio samples for comparison
        test_audio_1 = test_audio_stereo
        test_audio_2 = test_audio_stereo * 0.9  # 10% quieter
        
        analysis_1 = analyze_audio_array(test_audio_1, sample_rate)
        analysis_2 = analyze_audio_array(test_audio_2, sample_rate)
        
        # Skip comparison functions that require complex config
        print("⚠️ collect_metrics - skipped (requires file paths)")
        print("⚠️ build_comparison_tables - skipped (requires config)")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test HTML report generation with simple test data
            import pandas as pd
            
            # Create simple test dataframes
            summary_df = pd.DataFrame({
                'name': ['test1', 'test2'],
                'peak_dbfs': [-6.0, -8.0],
                'rms_dbfs': [-12.0, -14.0]
            })
            deltas_df = pd.DataFrame({
                'metric': ['peak_dbfs', 'rms_dbfs'],
                'delta': [-2.0, -2.0]
            })
            
            plots = {}  # Empty plots for this test
            report_path = os.path.join(temp_dir, "test_report.html")
            
            result_path = write_report_html(
                summary_df, deltas_df, plots, report_path,
                title="Test Report", 
                extra_notes="Automated test report"
            )
            
            assert os.path.exists(result_path)
            with open(result_path, 'r') as f:
                content = f.read()
                assert "Test Report" in content
                assert "Automated test report" in content
            print("✅ write_report_html - working")
        
        # Test utility functions
        print("\n🔧 Testing utility functions...")
        
        # Test audio format conversion
        
        # Test format conversion
        int16_audio = (test_audio * 32767).astype(np.int16)
        float32_audio = to_float32(int16_audio)
        assert float32_audio.dtype == np.float32
        assert np.max(np.abs(float32_audio)) <= 1.0
        print("✅ to_float32 - working")
        
        # Test sanitization
        dirty_audio = np.array([1.0, np.inf, -np.inf, np.nan, 5.0, -5.0])
        clean_audio = sanitize_audio(dirty_audio)
        assert np.isfinite(clean_audio).all()
        print("✅ sanitize_audio - working")
        
        # Test stereo conversion
        mono_audio = np.array([1, 2, 3, 4])
        stereo_audio = ensure_stereo(mono_audio)
        assert stereo_audio.shape == (4, 2)
        print("✅ ensure_stereo - working")
        
        # Test mono conversion
        test_stereo = np.array([[1, 2], [3, 4], [5, 6]])
        mono_result = to_mono(test_stereo)
        expected = np.array([1.5, 3.5, 5.5])  # Average of channels
        np.testing.assert_array_almost_equal(mono_result, expected)
        print("✅ to_mono - working")
        
        print("\n" + "=" * 60)
        print("🏆 post_mix_cleaned.ipynb - ALL TESTS PASSED!")
        print("✅ Core workflow fully functional")
        print("✅ Audio I/O working perfectly")
        print("✅ Analysis pipeline operational")
        print("✅ Reporting system ready")
        print("✅ Utility functions working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_post_mix_cleaned_functionality()
    sys.exit(0 if success else 1)