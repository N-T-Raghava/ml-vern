"""
Environment and dependency capture utilities.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def get_python_version() -> Dict[str, Any]:
    """Get detailed Python version information.

    Returns:
        Dict with version details
    """
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
        },
        "implementation": sys.implementation.name,
        "executable": sys.executable,
    }


def get_pip_freeze() -> List[str]:
    """Get pip freeze output as list of package strings.

    Returns:
        List of package specifications (e.g., ["numpy==1.23.0", ...])
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except Exception:
        return []


def get_installed_packages() -> Dict[str, str]:
    """Get dictionary of installed packages and versions.

    Returns:
        Dict mapping package name to version
    """
    packages = {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        pip_output = json.loads(result.stdout)
        packages = {pkg["name"]: pkg["version"] for pkg in pip_output}
    except Exception:
        pass
    return packages


def save_environment(run_path: str) -> Dict[str, Any]:
    """Capture and save environment information to run directory.

    Args:
        run_path: Path to the run directory

    Returns:
        Environment dict that was saved
    """
    os.makedirs(run_path, exist_ok=True)

    env_info = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": get_python_version(),
        "packages": get_installed_packages(),
        "pip_freeze": get_pip_freeze(),
    }

    env_path = os.path.join(run_path, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=4)

    return env_info


def load_environment(
    run_path: str,
) -> Optional[Dict[str, Any]]:
    """Load saved environment information from run directory.

    Args:
        run_path: Path to the run directory

    Returns:
        Environment dict, or None if file doesn't exist
    """
    env_path = os.path.join(run_path, "environment.json")

    if not os.path.exists(env_path):
        return None

    with open(env_path, "r") as f:
        return json.load(f)

        return json.load(f)
