import os
import shutil
import stat
import subprocess
import sys
from tempfile import mkstemp

from invoke import Collection, task


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()

        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def get_version(c):
    """
    Get version from setuptools-scm (git tags only - no fallbacks).

    Returns version string derived from git tags using setuptools-scm with guess-next-dev scheme.
    This means commits after a tag will get a version like "2.1.19.dev1" (next minor version).
    Raises RuntimeError with helpful message if version cannot be determined.
    """
    try:
        from setuptools_scm import get_version as scm_get_version

        return scm_get_version(
            root=os.path.dirname(__file__),
            version_scheme="guess-next-dev",
            local_scheme="no-local-version",
        )
    except Exception as e:
        print("ERROR: Unable to determine version from git tags.")
        print("Please ensure:")
        print("  1. setuptools-scm is installed (pip install setuptools-scm)")
        print("  2. You are in a git repository with at least one version tag")
        print("  3. Git tags follow the format 'v2.1.18' or similar")
        raise RuntimeError(f"Version determination failed: {e}")


# Handle long filenames or readonly files on windows, see:
# http://bit.ly/2g58Yxu
def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            try:
                os.remove(filename)
            except PermissionError:
                print(
                    "Permission error: unable to remove {}. Skipping that file.".format(
                        filename
                    )
                )
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                print(
                    "Unable to remove directory {}. Skipping removing "
                    "that folder.".format(os.path.join(root, name))
                )
    try:
        os.rmdir(top)
    except OSError:
        print(
            "Unable to remove directory {}. Skipping removing that folder.".format(top)
        )


# Function to find and replace in a file
def _replace(file_path, regex, subst):
    # Create temp file
    fh, abs_path = mkstemp()
    if sys.version_info[0] < 3:
        with os.fdopen(fh, "w") as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    new_file.write(regex.sub(subst, line))
    else:
        with open(fh, "w", encoding="Latin-1") as new_file:
            with open(file_path, encoding="Latin-1") as old_file:
                for line in old_file:
                    new_file.write(regex.sub(subst, line))
    os.remove(file_path)
    shutil.move(abs_path, file_path)


###############################################################################
# Misc development tasks (change version, deploy GEE scripts)
###############################################################################


@task
def set_version(c, version=None):
    """
    Generate _version.py with git information and update pyproject.toml dependencies.

    Args:
        version: Optional manual version string (e.g., "2.1.20"). If not provided,
                 version is determined automatically from git tags using setuptools-scm.

    Version behavior based on last digit:
    - Even number (e.g., 2.1.18): Stable release - use tagged versions for te_schemas
    - Odd number (e.g., 2.1.19): Development release - use master branch for te_schemas
    """
    import re

    # Get version - either from manual override or git tags
    if version:
        version_to_write = version
        print(f"Using manually specified version: {version_to_write}")
    else:
        version_to_write = get_version(c)
        print(f"Using version {version_to_write} from git tags")

    # Extract the last numeric component to determine if even or odd
    # Strip dev/post/rc suffixes (with or without dot separator)
    version_clean = re.sub(r"(\.?(dev|post|rc).*$)", "", version_to_write)
    version_parts = version_clean.split(".")
    try:
        last_number = int(version_parts[-1])
        is_even_version = (last_number % 2) == 0
    except (ValueError, IndexError):
        # If we can't parse, default to odd (experimental)
        is_even_version = False
        last_number = None

    if is_even_version:
        print(
            f"Even version detected ({last_number}) - Using tagged versions for dependencies"
        )
    else:
        print(
            f"Odd version detected ({last_number}) - Using master branch for dependencies"
        )

    # Always generate _version.py with git information captured at build time
    print("Generating te_algorithms/_version.py with git information")

    # Get git commit info at build time
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"

    try:
        git_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_date = "unknown"

    # Write _version.py with all information
    version_file_path = "te_algorithms/_version.py"
    with open(version_file_path, "w") as f:
        f.write("# This file is auto-generated at build time\n")
        f.write("# Do not edit this file manually\n")
        f.write(f'__version__ = "{version_to_write}"\n')
        f.write("\n")
        f.write("# Git information captured at build time\n")
        f.write(f'__git_sha__ = "{git_sha}"\n')
        f.write(f'__git_date__ = "{git_date}"\n')

    print(
        f"Successfully generated _version.py with version {version_to_write}, git SHA {git_sha}"
    )

    # Update pyproject.toml dependencies based on even/odd version
    print("Updating pyproject.toml dependencies")

    # Regex to match te_schemas dependency line
    # Handles both with and without @version, with optional trailing comma
    te_schemas_regex = re.compile(
        r'(\s*"te_schemas @ git\+https://github\.com/ConservationInternational/trends\.earth-schemas\.git)(@[.0-9a-z]+)?("(?:,)?)'
    )

    pyproject_path = "pyproject.toml"

    if is_even_version:
        # Stable release - use tagged version
        replacement = r"\g<1>@v" + version_clean + r"\g<3>"
        print(f"Setting te_schemas dependency to v{version_clean} in pyproject.toml")
    else:
        # Development release - use master branch
        replacement = r"\g<1>@master\g<3>"
        print("Setting te_schemas dependency to master in pyproject.toml")

    _replace(pyproject_path, te_schemas_regex, replacement)


###############################################################################
# Setup dependencies and install package
###############################################################################


def not_comments(lines, s, e):
    return [line for line in lines[s:e] if line[0] != "#"]


def read_requirements():
    """Return a list of runtime and list of test requirements"""
    with open("requirements.txt") as f:
        lines = f.readlines()
    lines = [line for line in [line.strip() for line in lines] if line]
    divider = "# test requirements"

    try:
        idx = lines.index(divider)
    except ValueError:
        raise Exception('Expected to find "{}" in requirements.txt'.format(divider))

    return not_comments(lines, 0, idx), not_comments(lines, idx + 1, None)


@task()
def set_tag(c, version=None):
    """
    Create and push a git tag for the current version.

    Args:
        version: Optional manual version string (e.g., "2.1.20"). If not provided,
                 version is determined automatically from git tags using setuptools-scm.
    """
    if version:
        v = version
        print(f"Using manually specified version: {v}")
    else:
        v = get_version(c)
        print(f"Using version {v} from git tags")

    ret = subprocess.run(
        ["git", "diff-index", "HEAD", "--"], capture_output=True, text=True
    )
    if ret.stdout != "":
        ret = query_yes_no("Uncommitted changes exist in repository. Commit these?")
        if ret:
            ret = subprocess.run(
                ["git", "commit", "-m", "Updating version tags for v{}".format(v)]
            )
            ret.check_returncode()
        else:
            print("Changes not committed - VERSION TAG NOT SET")

    print("Tagging version {} and pushing tag to origin".format(v))
    ret = subprocess.run(
        ["git", "tag", "-l", "v{}".format(v)], capture_output=True, text=True
    )
    ret.check_returncode()
    if "v{}".format(v) in ret.stdout:
        # Try to delete this tag on remote in case it exists there
        ret = subprocess.run(["git", "push", "origin", "--delete", "v{}".format(v)])
        if ret.returncode == 0:
            print("Deleted tag v{} on origin".format(v))
    subprocess.check_call(
        ["git", "tag", "-f", "-a", "v{}".format(v), "-m", "Version {}".format(v)]
    )
    subprocess.check_call(["git", "push", "origin", "v{}".format(v)])


###############################################################################
# Options
###############################################################################

ns = Collection(set_version, set_tag)
