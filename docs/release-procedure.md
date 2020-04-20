To update Modin version, use the following method (uses versioneer):

### Commit

        `git commit -a -m "Bump version to x.x.x"`

### Tag commit

        `git tag -as x.x.x`

  * Include release documentation in the annotation and make sure it is signed.
  * Push the tag to master: `git push upstream :refs/tags/x.x.x`

### Build wheels:

```bash
# Fresh clone Modin
git clone git@github.com:modin-project/modin.git
cd modin
# Build wheels. Wheels must be built per-distribution
SETUP_PLAT_NAME=macos python3 setup.py sdist bdist_wheel --plat-name macosx_10_9_x86_64
SETUP_PLAT_NAME=win32 python3 setup.py sdist bdist_wheel --plat-name win32
SETUP_PLAT_NAME=win_amd64 python3 setup.py sdist bdist_wheel --plat-name win_amd64
SETUP_PLAT_NAME=linux python3 setup.py sdist bdist_wheel --plat-name manylinux1_x86_64
SETUP_PLAT_NAME=linux python3 setup.py sdist bdist_wheel --plat-name manylinux1_i686
```

You may see the wheels in the `dist` folder: `ls -l dist`. Make sure the version is correct,
and make sure there are 5 distributions listed above with the `--plat-name` arguments.
Also make sure there is a `tar` file that contains the source.

### Upload wheels:

Use `twine` to upload wheels:

```bash
twine upload dist/*
```

### Check with `pip install`:

Run `pip install -U modin[all]` on Linux, Mac, and Windows systems in a new environment
to test that the wheels were uploaded correctly.
