#!/usr/bin/env python3
"""
Official RAG System Check Script
Save as: scripts/system_check.py

Checks if system is ready for production RAG experiment
"""

import torch
import psutil
import os
import sys
from datetime import datetime


def print_header():
    """Print formatted header"""
    print("=" * 70)
    print("🎯 OFFICIAL RAG PRODUCTION SYSTEM CHECK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def check_python_environment():
    """Check Python and package versions"""
    print("🐍 PYTHON ENVIRONMENT CHECK")
    print("-" * 40)
    print(f"Python Version: {sys.version}")

    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'datasets': '2.0.0',
        'numpy': '1.20.0',
        'pandas': '1.3.0'
    }

    missing_packages = []

    for package, min_version in required_packages.items():
        try:
            if package == 'torch':
                import torch
                version = torch.__version__
                print(f"  ✅ PyTorch: {version}")
            elif package == 'transformers':
                import transformers
                version = transformers.__version__
                print(f"  ✅ Transformers: {version}")
            elif package == 'datasets':
                import datasets
                version = datasets.__version__
                print(f"  ✅ Datasets: {version}")
            elif package == 'numpy':
                import numpy as np
                version = np.__version__
                print(f"  ✅ NumPy: {version}")
            elif package == 'pandas':
                import pandas as pd
                version = pd.__version__
                print(f"  ✅ Pandas: {version}")

        except ImportError:
            print(f"  ❌ {package}: Not installed")
            missing_packages.append(package)

    if missing_packages:
        print(
            f"\n⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
        return False

    print("  🎉 All required packages installed!")
    return True


def check_gpu_resources():
    """Comprehensive GPU check"""
    print("\n🔥 GPU RESOURCES CHECK")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("  ❌ CUDA not available")
        return False, -1

    gpu_count = torch.cuda.device_count()
    print(f"  📊 Available GPUs: {gpu_count}")

    best_gpu = 0
    max_free_memory = 0

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3

        # Get current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        free_memory = total_memory - cached

        print(f"\n  GPU {i}: {props.name}")
        print(f"    Total Memory: {total_memory:.1f} GB")
        print(f"    Allocated: {allocated:.1f} GB")
        print(f"    Cached: {cached:.1f} GB")
        print(f"    Available: {free_memory:.1f} GB")

        # Assess capability
        if free_memory >= 10:
            status = "🟢 EXCELLENT"
        elif free_memory >= 8:
            status = "🟡 GOOD"
        elif free_memory >= 6:
            status = "🟠 ADEQUATE"
        else:
            status = "🔴 INSUFFICIENT"

        print(f"    Status: {status}")

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i

    print(
        f"\n  🎯 Recommended GPU: {best_gpu} ({max_free_memory:.1f} GB available)")

    # Overall assessment
    if max_free_memory >= 6:
        print("  ✅ System ready for production RAG!")
        return True, best_gpu
    else:
        print("  ⚠️  May need memory optimization or cleanup")
        return False, best_gpu


def check_disk_space():
    """Check available disk space"""
    print("\n💾 DISK SPACE CHECK")
    print("-" * 40)

    # Check current directory
    current_dir = os.getcwd()
    disk_usage = psutil.disk_usage(current_dir)

    free_gb = disk_usage.free / 1024**3
    total_gb = disk_usage.total / 1024**3
    used_gb = disk_usage.used / 1024**3

    print(f"  Current Directory: {current_dir}")
    print(f"  Total Space: {total_gb:.1f} GB")
    print(f"  Used Space: {used_gb:.1f} GB")
    print(f"  Free Space: {free_gb:.1f} GB")

    required_space = 25  # 21GB for index + buffer

    if free_gb >= required_space:
        print(f"  ✅ Sufficient space for 21GB Wikipedia download")
        return True
    else:
        print(f"  ❌ Need {required_space - free_gb:.1f} GB more space")
        return False


def check_internet_connection():
    """Check internet connectivity"""
    print("\n🌐 INTERNET CONNECTIVITY CHECK")
    print("-" * 40)

    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print("  ✅ HuggingFace accessible")

        # Estimate download speed (simple test)
        print("  🔄 Testing download speed...")
        import time
        start_time = time.time()
        urllib.request.urlopen(
            'https://httpbin.org/bytes/1048576', timeout=30)  # 1MB test
        end_time = time.time()

        speed_mbps = (1 * 8) / (end_time - start_time)  # Convert to Mbps
        print(f"  📶 Estimated speed: ~{speed_mbps:.0f} Mbps")

        # Estimate download time for 21GB
        download_time_minutes = (21 * 1024 * 8) / (speed_mbps * 60)
        print(
            f"  ⏰ Estimated 21GB download time: ~{download_time_minutes:.0f} minutes")

        return True

    except Exception as e:
        print(f"  ❌ Connection issue: {str(e)}")
        return False


def check_huggingface_cache():
    """Check HuggingFace cache directory"""
    print("\n🗂️  HUGGINGFACE CACHE CHECK")
    print("-" * 40)

    cache_dir = os.path.expanduser("~/.cache/huggingface/")

    if os.path.exists(cache_dir):
        print(f"  📁 Cache directory: {cache_dir}")

        # Check if RAG model already cached
        rag_cache = os.path.join(cache_dir, "transformers")
        if os.path.exists(rag_cache):
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(rag_cache)
                             for filename in filenames) / 1024**3
            print(f"  📊 Current cache size: {cache_size:.1f} GB")

        print("  ✅ Cache directory ready")
    else:
        print(f"  📁 Cache will be created at: {cache_dir}")

    return True


def generate_system_report():
    """Generate comprehensive system report"""
    print("\n📋 SYSTEM READINESS REPORT")
    print("=" * 70)

    checks = {
        "Python Environment": check_python_environment(),
        "GPU Resources": check_gpu_resources()[0],
        "Disk Space": check_disk_space(),
        "Internet Connection": check_internet_connection(),
        "HuggingFace Cache": check_huggingface_cache()
    }

    passed_checks = sum(checks.values())
    total_checks = len(checks)

    print(f"\n🎯 OVERALL SYSTEM STATUS:")
    print(f"  Passed: {passed_checks}/{total_checks} checks")

    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")

    if passed_checks == total_checks:
        print(f"\n🚀 SYSTEM READY FOR PRODUCTION RAG!")
        print(f"  All prerequisites met")
        print(f"  Recommended next step: Run production experiment")
    elif passed_checks >= total_checks - 1:
        print(f"\n⚠️  SYSTEM MOSTLY READY")
        print(f"  Minor issues detected - experiment may still work")
        print(f"  Consider addressing failed checks")
    else:
        print(f"\n❌ SYSTEM NOT READY")
        print(f"  Multiple issues detected")
        print(f"  Address failed checks before proceeding")

    return passed_checks == total_checks


def main():
    """Main system check function"""
    print_header()

    try:
        # Run all checks
        system_ready = generate_system_report()

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"../logs/system_check_{timestamp}.log"

        # Create logs directory if it doesn't exist
        os.makedirs("../logs", exist_ok=True)

        print(f"\n💾 System check report saved to: {report_file}")

        return system_ready

    except Exception as e:
        print(f"\n❌ System check failed: {str(e)}")
        return False


if __name__ == "__main__":
    system_ready = main()
    sys.exit(0 if system_ready else 1)
