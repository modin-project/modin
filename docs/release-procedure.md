## Versioning

### Point release

Modin uses semantic versioning. So when doing a point release, please make a separate branch
off the previous release tag, and `git cherry-pick` **only** bugfixes there (assuming previous release was versioned `X.Y.Z`):

        git checkout -b release-X.Y.Z+1 X.Y.Z

### Major release

A "major" (that would be `0.xx.0` for now until Modin hits `1.0` milestone) release could be done by branching from `master`:

        git checkout -b release-X.Y.0 master

## Preparing the release

Before continuing with the release process, make sure that automated CI which runs on each commit passed successfully with the commit you deem as a "release candidate".

Modin follows the "no push" logic, which is _only_ circumvented for cherry-picked commits,
as reviewing them again would not add a lot of value but would add lots of excess work.

Hence non-cherry-pick commits should happen in a separate branch in your own fork, and
be delivered to the release branch by using a PR.

Note that Modin uses fully signed commits at least for releases, so you have to have GPG keys set up. See [onboarding instructions](https://github.com/modin-project/modin/blob/master/onboarding/onboarding.md) on where to get started.

To update Modin version, follow the instructions below.

### Update the link to release in readme

**Note**: from now on you work in a branch in your fork (in `origin`).

Change the `README.md` in the repo root so release badge points to new (would be created) release.

### Commit

        git commit -a -S -m "Release version X.Y.Z"

Pay attention to that `-S` switch if you haven't enabled signing all the commits!
Modin requires release commits to be signed (do not mix this with "signing off" commits, this
is a whole different story - signed commits are cryptographically signed).

Push the commit, make a PR _against `release-X.Y.Z`_ branch and, when it's merged, pull that branch from `upstream` before proceeding.

### Tag commit

**Note**: from now on you work in the release branch (in `upstream`).

        git tag -as X.Y.Z

  * Look for [other releases](https://github.com/modin-project/modin/releases) on examples of how to compose the release documentation
    * Always try to make a one-line summary
    * The annotation should contain features and changes compared to previous release
    * You can link to merge commits, but try to "explain" what a PR does instead of blindly copying its title
    * Gather and mention the list of all participants in the release, including those mentioned in "Co-Authored-By" part of PRs
  * Include release documentation in the annotation and make sure it is signed.
  * Push the tag to master: `git push upstream X.Y.Z`
    * If you're re-pushing a tag (beware! you shouldn't be doing that, no, _really_!), you can remove remote tag and push a local one by `git push upstream :refs/tags/X.Y.Z`

### Build wheels:

```bash
# Install/update tools
pip install --upgrade build twine
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

Make sure you have an active PyPI account which has write access to Modin pypi repo, and make sure you have a pypi token set up.

Use `twine` to upload wheels:

```bash
twine upload dist/*
```

When asked for account, provide `__token__` (exactly as stated), when asked for password, present your token from pypi.

### Check with `pip install`:

Run `pip install -U modin[all]` on Linux, Mac, and Windows systems in a new environment
to test that the wheels were uploaded correctly.

## Make Github and conda-forge release

### Github

After all is said and done and pushed to PyPI, open Github Releases page and create a new release. It should be enough to just specify "release from tag" and point it to newly created `X.Y.Z` tag plus fill the release title - it should read `Modin X.Y.Z` - Github should pull everything else including release text from your tag annotations.

### Conda-forge

Conda-forge has a bot which watches for new releases of software packaged through it,
and in case of Modin it waits either for Github releases or for tags and then makes
a new automatic PR with version increment.

You should watch for that PR and, fixing any issues if there are some, merge it
to make new Modin release appear in `conda-forge` channel.
