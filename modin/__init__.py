import os
import subprocess


def git_version():
    def _execute_cmd_in_temp_env(cmd):
        # construct environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]

    try:
        git_revision = _execute_cmd_in_temp_env(['git', 'rev-parse', 'HEAD'])
        return git_revision.strip().decode('ascii')
    except OSError:
        return "Unknown"


__git_revision__ = git_version()
__version__ = "0.1.2"
