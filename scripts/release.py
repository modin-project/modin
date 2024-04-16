import argparse
import atexit
import collections
import json
import re
import sys
from pathlib import Path

import github
import pygit2
from packaging import version


class GithubUserResolver:
    def __init__(self, email2commit, token):
        self.__cache_file = Path(__file__).parent / "gh-users-cache.json"
        self.__cache = (
            json.loads(self.__cache_file.read_text())
            if self.__cache_file.exists()
            else {}
        )
        # filter unknown users hoping we'd be able to find them this time
        self.__cache = {key: value for key, value in self.__cache.items() if value}
        # using anonymous access if token not specified
        self.__github = github.Github(token or None)
        self.__modin_repo = self.__github.get_repo("modin-project/modin")
        self.__email2commit = email2commit
        atexit.register(self.__save)

    def __search_commits(self, term):
        if commit := self.__email2commit.get(term):
            gh_commit = self.__modin_repo.get_commit(str(commit))
            return gh_commit.author.login
        return None

    @staticmethod
    def __is_email(term):
        return re.match(r".*@.*\..*", term)

    def __search_github(self, term):
        search = f"in:email {term}" if self.__is_email(term) else f"fullname:{term}"
        match = [user.login for user in self.__github.search_users(search)]
        return match[0] if len(match) == 1 else None

    def __try_user(self, term):
        if self.__is_email(term):
            return None
        try:
            return self.__github.get_user(term).login
        except github.GithubException as ex:
            if ex.status != 404:
                raise
            return None

    def __resolve_single(self, term):
        return (
            self.__search_commits(term)
            or self.__search_github(term)
            or self.__try_user(term)
        )

    def __resolve_cache(self, name, email):
        return self.__cache.get(f"{name} <{email}>", None)

    def __register(self, name, email, match):
        self.__cache[f"{name} <{email}>"] = match

    def resolve(self, people):
        logins, unknowns = set(), set()

        for name, email in people:
            if match := self.__resolve_cache(name, email):
                logins.add(match)
            elif match := self.__resolve_single(email):
                self.__register(name, email, match)
                logins.add(match)
            else:
                if match := self.__resolve_single(name):
                    logins.add(match)
                else:
                    unknowns.add((name, email))
                self.__register(name, email, match)

        return logins, unknowns

    def resolve_by_reviews(self, unknowns, email2pr):
        logins, new_unknowns = set(), set()
        for name, email in unknowns:
            commit = self.__modin_repo.get_commit(str(email2pr[email]))
            found = set()
            for pull in commit.get_pulls():
                for review in pull.get_reviews():
                    user = review.user
                    if user.name == name and (not user.email or user.email == email):
                        found.add(user.login)

            if len(found) == 1:
                self.__register(name, email, list(found)[0])
                logins |= found
            else:
                new_unknowns.add((name, email))

        return logins, new_unknowns

    def __save(self):
        self.__cache_file.write_text(json.dumps(self.__cache, indent=4, sort_keys=True))


class GitWrapper:
    def __init__(self):
        self.repo = pygit2.Repository(Path(__file__).parent)

    def is_on_main(self):
        return self.repo.references["refs/heads/main"] == self.repo.head

    @staticmethod
    def __get_tag_version(entry):
        try:
            return version.parse(entry.lstrip("refs/tags/"))
        except version.InvalidVersion as ex:
            return f'<bad version "{entry}": {ex}>'

    def get_previous_release(self, rel_type):
        tags = [
            (entry, self.__get_tag_version(entry))
            for entry in self.repo.references
            if entry.startswith("refs/tags/")
        ]
        # filter away legacy versions (which aren't following the proper naming schema);
        # also skip pre-releases
        tags = [
            (entry, ver)
            for entry, ver in tags
            if isinstance(ver, version.Version) and not ver.pre
        ]
        if rel_type == "minor":
            # leave only minor releases
            tags = [(entry, ver) for entry, ver in tags if ver.micro == 0]
        else:
            assert rel_type == "patch"
        prev_ref, prev_ver = max(tags, key=lambda pair: pair[1])
        return prev_ref, self.repo.references[prev_ref].peel(), prev_ver

    def get_commits_upto(self, stop_commit):
        history = []
        for obj in self.repo.walk(self.repo.head.target):
            if obj.id == stop_commit.id:
                break
            history.append(obj)
        else:
            raise ValueError("Current HEAD is not derived from previous release")
        return history

    def ensure_title_link(self, obj: pygit2.Commit):
        title = obj.message.splitlines()[0]
        if not re.match(r".*\(#(\d+)\)$", title):
            title += f" ({obj.short_id})"
        return title


def make_notes(args):
    wrapper = GitWrapper()
    release_type = "minor" if wrapper.is_on_main() else "patch"
    sys.stderr.write(f"Detected release type: {release_type}\n")

    prev_ref, prev_commit, prev_ver = wrapper.get_previous_release(release_type)
    sys.stderr.write(f"Previous {release_type} release: {prev_ref}\n")

    next_major, next_minor, next_patch = prev_ver.release
    if release_type == "minor":
        next_minor += 1
    elif release_type == "patch":
        next_patch += 1
    else:
        raise ValueError(f"Unexpected release type: {release_type}")
    next_ver = version.Version(f"{next_major}.{next_minor}.{next_patch}")

    sys.stderr.write(f"Computing release notes for {prev_ver} -> {next_ver}...\n")
    try:
        history = wrapper.get_commits_upto(prev_commit)
    except ValueError as ex:
        sys.stderr.write(
            f"{ex}: did you forget to checkout correct branch or pull tags?"
        )
        return 1
    if not history:
        sys.stderr.write(f"No commits since {prev_ver} found, nothing to generate!\n")
        return 1

    titles = collections.defaultdict(list)
    people = set()
    email2commit, email2pr = {}, {}
    for obj in history:
        title = obj.message.splitlines()[0]
        titles[title.split("-")[0]].append(obj)
        new_people = set(
            re.findall(
                r"(?:(?:Signed-off-by|Co-authored-by):\s*)([\w\s,]+?)\s*<([^>]+)>",
                obj.message,
            )
        )
        for _, email in new_people:
            email2pr[email] = obj.id
        people |= new_people
        email2commit[obj.author.email] = obj.id
    sys.stderr.write(f"Found {len(history)} commit(s) since {prev_ref}\n")

    sys.stderr.write("Resolving contributors...\n")
    user_resolver = GithubUserResolver(email2commit, args.token)
    logins, unknowns = user_resolver.resolve(people)
    new_logins, unknowns = user_resolver.resolve_by_reviews(unknowns, email2pr)
    logins |= new_logins
    sys.stderr.write(f"Found {len(logins)} GitHub usernames.\n")
    if unknowns:
        sys.stderr.write(
            f"Warning! Failed to resolve {len(unknowns)} usernames, please resolve them manually!\n"
        )

    sections = [
        ("Stability and Bugfixes", "FIX"),
        ("Performance enhancements", "PERF"),
        ("Refactor Codebase", "REFACTOR"),
        ("Update testing suite", "TEST"),
        ("Documentation improvements", "DOCS"),
        ("New Features", "FEAT"),
    ]

    notes = rf"""Modin {next_ver}

<Please fill in short release summary>

Key Features and Updates Since {prev_ver}
-------------------------------{'-' * len(str(prev_ver))}
"""

    def _add_section(section, prs):
        nonlocal notes
        if prs:
            notes += f"* {section}\n"
            notes += "\n".join(
                [
                    f"  * {wrapper.ensure_title_link(obj)}"
                    for obj in sorted(prs, key=lambda obj: obj.message)
                ]
            )
            notes += "\n"

    for section, key in sections:
        _add_section(section, titles.pop(key, None))

    uncategorized = sum(titles.values(), [])
    _add_section("Uncategorized improvements", uncategorized)

    notes += r"""
Contributors
------------
"""
    notes += "\n".join(f"@{login}" for login in sorted(logins)) + "\n"
    notes += (
        "\n".join(
            f"<unknown-login> {name} <{email}>" for name, email in sorted(unknowns)
        )
        + "\n"
    )

    sys.stdout.write(notes)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--token",
        type=str,
        default="",
        help="GitHub token for queries (optional, bumps up rate limit)",
    )
    parse.set_defaults(func=lambda _: parse.print_usage())
    subparsers = parse.add_subparsers()

    notes = subparsers.add_parser("notes", help="Generate release notes")
    notes.set_defaults(func=make_notes)

    args = parse.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
