"""
setup.py — Install dependencies and start Qdrant via Docker for activity7.
Run: python setup.py
"""

import subprocess
import sys
import time


def run(cmd, check=True):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and check:
        print(result.stderr.strip())
    return result


def main():
    print("=" * 50)
    print("  Activity 7 — Setup")
    print("=" * 50)

    # Install dependencies
    print("\n[1] Installing Python dependencies...")
    run(f"{sys.executable} -m pip install qdrant-client python-dotenv")

    # Check if Docker is available
    print("\n[2] Checking Docker...")
    docker_check = run("docker --version", check=False)

    if docker_check.returncode != 0:
        print("  Docker not found. Please install Docker Desktop and re-run.")
        print("  https://www.docker.com/products/docker-desktop/")
        print("\n  Alternatively, use Qdrant Cloud (free) at https://cloud.qdrant.io")
        print("  Then set QDRANT_URL and QDRANT_API_KEY in your .env file.")
        return

    # Start Qdrant container
    print("\n[3] Starting Qdrant container...")
    run("docker rm -f qdrant-activity7 2>nul || true", check=False)
    run(
        "docker run -d --name qdrant-activity7 "
        "-p 6333:6333 "
        "qdrant/qdrant"
    )

    print("\n  Waiting for Qdrant to start...")
    time.sleep(4)

    # Verify
    print("\n[4] Verifying Qdrant is running...")
    check = run("docker ps --filter name=qdrant-activity7 --format table", check=False)
    print("\nSetup complete! Now run:  python quadrant.py")


if __name__ == "__main__":
    main()
