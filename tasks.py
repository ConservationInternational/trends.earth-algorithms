# -*- coding: utf-8 -*-

import os
import sys
import re
import shutil
import subprocess
from tempfile import mkstemp
import json
from datetime import datetime, timezone

from invoke import Collection, task


def get_version(c):
    with open(c.version_file_raw, 'r') as f:
        return f.readline().strip()

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
                print('Permission error: unable to remove {}. Skipping that file.'.format(filename))
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                print('Unable to remove directory {}. Skipping removing that folder.'.format(os.path.join(root, name)))
    try:
        os.rmdir(top)
    except OSError:
        print('Unable to remove directory {}. Skipping removing that folder.'.format(top))


# Function to find and replace in a file
def _replace(file_path, regex, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    if sys.version_info[0] < 3:
        with os.fdopen(fh,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    new_file.write(regex.sub(subst, line))
    else:
        with open(fh, 'w', encoding='Latin-1') as new_file:
            with open(file_path, encoding='Latin-1') as old_file:
                for line in old_file:
                    new_file.write(regex.sub(subst, line))
    os.remove(file_path)
    shutil.move(abs_path, file_path)


###############################################################################
# Misc development tasks (change version, deploy GEE scripts)
###############################################################################

@task(help={'v': 'Version to set'})
def set_version(c, v=None):
    # Validate the version matches the regex
    if not v:
        version_update = False
        v = get_version(c)
        print('No version specified, retaining version {}, but updating SHA and release date'.format(v))
    elif not re.match("[0-9]+([.][0-9]+)+", v):
        print('Must specify a valid version (example: 0.36)')
        return
    else:
        version_update = True
    
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip('\n')[0:8]
    release_date = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%SZ')

    # Set in version.json
    print('Setting version to {} in version.json'.format(v))
    with open(c.version_file_details, 'w') as f:
        json.dump({"version": v, "revision": revision, "release_date": release_date}, f,  indent=4)

    if version_update:
        # Set in version.txt
        print('Setting version to {} in {}'.format(v, c.version_file_raw))
        with open(c.version_file_raw, 'w') as f:
            f.write(v)

        # Set in setup.py
        print('Setting version to {} in setup.py'.format(v))
        setup_regex = re.compile("^([ ]*version=[ ]*')[0-9]+([.][0-9]+)+")
        _replace('setup.py', setup_regex, '\g<1>' + v)


###############################################################################
# Setup dependencies and install package
###############################################################################

def read_requirements():
    """Return a list of runtime and list of test requirements"""
    with open('requirements.txt') as f:
        lines = f.readlines()
    lines = [ l for l in [ l.strip() for l in lines] if l ]
    divider = '# test requirements'

    try:
        idx = lines.index(divider)
    except ValueError:
        raise BuildFailure(
            'Expected to find "{}" in requirements.txt'.format(divider))

    not_comments = lambda s,e: [ l for l in lines[s:e] if l[0] != '#']
    return not_comments(0, idx), not_comments(idx+1, None)

###############################################################################
# Options
###############################################################################

ns = Collection(set_version)

ns.configure({
    'version_file_raw': 'version.txt',
    'version_file_details': 'version.json',
})
