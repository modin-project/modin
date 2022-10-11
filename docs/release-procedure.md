## Versioning

### Patch release

Modin uses semantic versioning. So when doing a patch release, please make a separate branch
off the previous release tag, and `git cherry-pick` **only** the commits we would like to have in our
patch release (assuming previous release was versioned `X.Y.Z`):

        git checkout -b release-X.Y.Z+1 X.Y.Z

### Major and Minor releases

A "major" (that would be `0.xx.0` for now until Modin hits `1.0` milestone) release could be done by branching from `master`:

        git checkout -b release-X.Y.0 master

## Preparing the release

Before continuing with the release process, make sure that automated CI which runs on each commit passed successfully with the commit you deem as a "release candidate".

Modin follows the "no push" logic, which is _only_ circumvented for cherry-picked commits,
as reviewing them again would not add a lot of value but would add lots of excess work.

Hence non-cherry-pick commits should happen in a separate branch in your own fork, and
be delivered to the release branch by using a PR.

Note that Modin uses fully signed commits, so you have to have GPG keys set up. See [onboarding instructions](https://github.com/modin-project/modin/blob/master/onboarding/onboarding.md) on where to get started.

To update Modin version, follow the instructions below.

### Preparing the repo for a Major or Minor Version

**Note**: this should be done in your fork of Modin.

First, update your fork of Modin's master with the main repo's master. From your master, create a new
branch called `release-X.Y.0` off of master. Create an empty commit in your new branch with the message
`Release version X.Y.0`. Make sure to sign this commit with both your GPG key
and with the conventional `git commit -s` (so `git commit -s -S`). Open a PR against modin-project/modin with just this commit.

### Preparing the repo for a Patch Version

**Note**: this should be done in the original Modin repository (in `upstream`) .

First, you must create a new branch in the upstream (main modin-project/modin) repo for the new release.
This branch must be named `release-X.Y.Z`, and should be made off of the tag for the last release. To
do this, look up the commit that corresponds to the last release, copy its SHA, and then do 
`git checkout <PREVIOUS_RELEASE_COMMIT_SHA>`. Next, do `git checkout -b release-X.Y.Z` to create
the branch for the new release, and push it upstream.

**Note**: now you must switch to your fork of Modin.

From your fork of Modin, fetch the upstream repo, and checkout the release branch you made above.
From this release branch, create a new branch, and cherry-pick the commits that will go into this release.

Once these commits have been applied to your branch, edit the `README.md` so that the PyPi badge will
point to the badge for this specific version (instead of latest) and so that the docs link will point
to the docs for this specific version (rather than latest).

Once this has been committed, create an empty commit, the same as for a major or minor version,
with the message `Release version X.Y.Z`, and make sure to sign it with both your GPG key, and the
traditional git sign-off. Create a PR using your branch against the `release-X.Y.Z` branch in the
original Modin repo.

### Tag commit

After the PR has been merged, clone a clean copy of the Modin repo from the modin-project organization.
You now need to tag the commit that corresponds to the above PR with the appropriate tag for this release.

**Note**: from now on you work on the master branch (in `upstream`) .

        git tag -as X.Y.Z

  * Look for [other releases](https://github.com/modin-project/modin/releases) on examples of how to compose the release documentation
    * Start with a `Modin X.Y.Z` line followed by an empty line
    * Always try to make a one-line summary
    * The annotation should contain features and changes compared to previous release
    * You can link to merge commits, but try to "explain" what a PR does instead of blindly copying its title
    * Gather and mention the list of all participants in the release, including those mentioned in "Co-Authored-By" part of PRs
  * Include release documentation in the annotation and make sure it is signed.
  * Push the tag to master: `git push upstream X.Y.Z`
    * If you're re-pushing a tag (beware! you shouldn't be doing that, no, _really_!), you can remove remote tag and push a local one by `git push upstream :refs/tags/X.Y.Z`

***Note***: We are currently working on automating the release notes procedure - check 
[#4747](https://github.com/modin-project/modin/issues/4747) for updates!

### Build wheel:

**Note**: This should be done from your clean fork of Modin from the modin-project organization, where
you made the release tag.

```bash
# Install/update tools
pip install --upgrade build twine
# Build a pure Python wheel.
python3 setup.py sdist bdist_wheel
```

You may see the wheel in the `dist` folder: `ls -l dist`. Make sure the version is correct.
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

Once the tag has been published, we need to make the release on GitHub. Go to the
[Release page](https://github.com/modin-project/modin/releases), and click on `Draft a new release`.
Choose the tag you made above from the dropdown menu, and copy paste the name of the release 
in the `Release title` box. Next, copy paste the release notes from above into the box labelled
`Describe this release`. This will ensure that the release notes on GitHub are Markdown formatted.

Double check that everything looks good by clicking `Preview`, and then hit the green `Publish release`
button!

### Conda-forge

Conda-forge has a bot which watches for new releases of software packaged through it,
and in case of Modin it waits either for Github releases or for tags and then makes
a new automatic PR with version increment.

You should watch for that PR and, fixing any issues if there are some, merge it
to make new Modin release appear in `conda-forge` channel.
