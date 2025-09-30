import argparse
import contextlib
import datetime
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Iterator

package = "keras_cv"
dist_directory = "dist"
wheels_directory = "wheels"


def run_command(cmd, env=None, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def parse_version(version_file: pathlib.Path) -> str:
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', content)
    if not match:
        raise ValueError("Unable to parse __version__ from version_utils.py")
    return match.group(1)


@contextlib.contextmanager
def nightly_overrides(root: pathlib.Path, is_nightly: bool) -> Iterator[str]:
    version_file = root / package / "src" / "version_utils.py"
    setup_file = root / "setup.py"

    original_version_content = version_file.read_text()
    original_setup_content = setup_file.read_text()
    version = parse_version(version_file)
    updated_version = version

    try:
        if is_nightly:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H")
            updated_version = f"{version}.dev{timestamp}"
            new_version_content = re.sub(
                r'__version__\s*=\s*["\'](.+?)["\']',
                f'__version__ = "{updated_version}"',
                original_version_content,
            )
            version_file.write_text(new_version_content)
            setup_file.write_text(
                original_setup_content.replace(
                    'name="keras-cv"', 'name="keras-cv-nightly"'
                )
            )
        yield updated_version
    finally:
        if is_nightly:
            version_file.write_text(original_version_content)
            setup_file.write_text(original_setup_content)


def locate_wheel(wheels_path: pathlib.Path, version: str) -> pathlib.Path:
    if not wheels_path.exists():
        return None
    wheels = sorted(
        wheels_path.glob("*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for wheel in wheels:
        if version in wheel.name:
            return wheel
    return wheels[0] if wheels else None


def build(root_path: pathlib.Path, is_nightly: bool) -> pathlib.Path:
    root_path = root_path.resolve()
    wheels_path = root_path / wheels_directory
    wheels_path.mkdir(parents=True, exist_ok=True)

    with nightly_overrides(root_path, is_nightly) as version:
        cwd = str(root_path)
        run_command(["python3", "build_deps/configure.py"], cwd=cwd)
        run_command(["bazel", "build", "build_pip_pkg"], cwd=cwd)
        env = os.environ.copy()
        env["BUILD_WITH_CUSTOM_OPS"] = "true"
        run_command(
            [str(root_path / "bazel-bin" / "build_pip_pkg"), wheels_directory],
            env=env,
            cwd=cwd,
        )

    wheel = locate_wheel(wheels_path, version)
    if not wheel:
        print("Build failed. No wheel found.")
        return None

    dist_path = root_path / dist_directory
    dist_path.mkdir(parents=True, exist_ok=True)
    target = dist_path / wheel.name
    shutil.copy2(wheel, target)
    print(f"Build successful. Wheel file available at {target.resolve()}")
    return target.resolve()


def install_whl(whl_path: pathlib.Path):
    run_command(
        [
            "pip3",
            "install",
            "--force-reinstall",
            "--no-dependencies",
            str(whl_path),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="Whether to install the generated wheel file.",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Whether to generate nightly wheel file.",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).parent.resolve()
    try:
        wheel_path = build(root, args.nightly)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)

    if wheel_path and args.install:
        install_whl(wheel_path)
