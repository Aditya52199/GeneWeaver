
import argparse
import importlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def decode_device_name(name):
    """Convert a CUDA device name into readable text."""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")

    return str(name)


def detect_cuda():
    """
    Detect Numba CUDA availability without making Numba
    a mandatory dependency for CPU-only users.
    """

    report = {
        "project": "GeneWeaver",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "operating_system": platform.platform(),
            "python_version": platform.python_version()
        },
        "numba": {
            "installed": False,
            "version": None
        },
        "cuda": {
            "available": False,
            "gpu_name": None,
            "compute_capability": None,
            "total_memory_mb": None,
            "multiprocessor_count": None,
            "status": "Not checked"
        },
        "execution_mode": "CPU",
        "message": ""
    }

    try:
        report["numba"]["version"] = importlib.metadata.version("numba")
        report["numba"]["installed"] = True

    except importlib.metadata.PackageNotFoundError:
        report["message"] = (
            "Numba is not installed. CPU mode is available. "
            "Install the optional GPU dependency to check CUDA."
        )
        report["cuda"]["status"] = "Numba not installed"
        return report

    try:
        cuda = importlib.import_module("numba.cuda")

    except Exception as error:
        report["cuda"]["status"] = "Numba CUDA module could not be loaded"
        report["message"] = (
            f"CUDA module import failed: {type(error).__name__}: {error}"
        )
        return report

    try:
        if not cuda.is_available():
            report["cuda"]["status"] = "CUDA device unavailable"
            report["message"] = (
                "Numba is installed, but no usable CUDA device was detected. "
                "Check your NVIDIA GPU, driver and CUDA compatibility."
            )
            return report

        device = cuda.get_current_device()

        report["cuda"]["available"] = True
        report["cuda"]["gpu_name"] = decode_device_name(
            getattr(device, "name", "Unknown")
        )

        capability = getattr(device, "compute_capability", None)

        if capability is not None:
            report["cuda"]["compute_capability"] = list(capability)

        total_memory = getattr(device, "total_memory", None)

        if total_memory is not None:
            report["cuda"]["total_memory_mb"] = round(
                total_memory / (1024 * 1024), 2
            )

        multiprocessors = getattr(
            device,
            "MULTIPROCESSOR_COUNT",
            None
        )

        if multiprocessors is not None:
            report["cuda"]["multiprocessor_count"] = int(
                multiprocessors
            )

        report["cuda"]["status"] = "CUDA device detected"
        report["execution_mode"] = "CUDA-capable"
        report["message"] = (
            "A CUDA device was detected. GPU kernel execution "
            "should still be tested before using it for production scans."
        )

    except Exception as error:
        report["cuda"]["status"] = "CUDA device inspection failed"
        report["message"] = (
            f"CUDA inspection failed: {type(error).__name__}: {error}"
        )

    return report


def print_report(report):
    print("\nGeneWeaver - GPU Environment Check")
    print("=" * 55)

    print(
        f"Operating system : "
        f"{report['system']['operating_system']}"
    )
    print(
        f"Python version   : "
        f"{report['system']['python_version']}"
    )

    print(
        f"Numba installed  : "
        f"{report['numba']['installed']}"
    )
    print(
        f"Numba version    : "
        f"{report['numba']['version'] or 'Not installed'}"
    )

    print(
        f"CUDA available   : "
        f"{report['cuda']['available']}"
    )
    print(
        f"CUDA status      : "
        f"{report['cuda']['status']}"
    )

    if report["cuda"]["available"]:
        print(
            f"GPU model        : "
            f"{report['cuda']['gpu_name']}"
        )
        print(
            f"Compute capability: "
            f"{report['cuda']['compute_capability']}"
        )
        print(
            f"GPU memory       : "
            f"{report['cuda']['total_memory_mb']} MB"
        )
        print(
            f"Multiprocessors  : "
            f"{report['cuda']['multiprocessor_count']}"
        )

    print(f"Execution mode   : {report['execution_mode']}")
    print(f"Message          : {report['message']}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Check GeneWeaver CUDA and GPU environment."
    )

    parser.add_argument(
        "--output",
        default="gpu_environment_report.json",
        help="JSON report output path"
    )

    args = parser.parse_args()

    report = detect_cuda()
    print_report(report)

    output_path = Path(args.output)

    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print(f"\nEnvironment report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
