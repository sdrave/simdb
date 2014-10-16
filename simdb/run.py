# -*- coding: utf-8 -*-
# Copyright (c) 2014, Stephan Rave
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import, division, print_function

import atexit
from cPickle import dump
import datetime
import os
import socket
import sys
import time
import traceback
import warnings
import __main__

import yaml

from simdb.db import DataLoader


# Customize YAML serialization
import numpy as np
def ndarray_representer(dumper, data):
    data = data.tolist()
    if isinstance(data, list):
        return dumper.represent_list(data)
    else:
        return dumper.represent_float(data)

def numpy_scalar_representer(dumper, data):
    return dumper.represent_float(float(data))

def unicode_representer(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')
    else:
        return dumper.represent_unicode(data)

def str_representer(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')
    else:
        return dumper.represent_str(data)

yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_representer(np.float64, numpy_scalar_representer)
yaml.add_representer(unicode, unicode_representer)
yaml.add_representer(str, str_representer)


_db_path = None
_run = None
_current_dataset = None
_current_dataset_data = None


def _initialize():
    global _run, _db_path
    try:
        db_path = os.environ['SIMDB_PATH']
    except KeyError:
        raise RuntimeError('SIMDB_PATH is not set!')

    try:
        git_repos = os.environ['SIMDB_GIT_REPOS']
    except KeyError:
        git_repos = ''
        warnings.warn('No git repositories specified via SIMDB_GIT_REPOS')

    if git_repos:
        git_repos = git_repos.split(':')
        git_info = {p.rstrip('/').split('/')[-1]: _git_info(p) for p in git_repos}
    else:
        git_info = None

    try:
        script = os.path.abspath(__main__.__file__)
    except AttributeError:
        script = 'from console'
    try:
        script_content = open(__main__.__file__, 'r').read()
    except AttributeError:
        script_content = ''
    argv = sys.argv

    host = os.uname()
    started = datetime.datetime.now()

    if not os.path.exists(os.path.join(db_path, 'RUNS')):
        os.mkdir(os.path.join(db_path, 'RUNS'))
    uid = _make_uid(os.path.join(db_path, 'RUNS'))
    os.mkdir(os.path.join(db_path, 'RUNS', uid))
    yaml.dump(dict(script=script,
                   argv=argv,
                   host=host,
                   git=git_info,
                   started=started),
              open(os.path.join(db_path, 'RUNS', uid, 'INFO'), 'w'),
              allow_unicode=True)
    with open(os.path.join(db_path, 'RUNS', uid, 'SCRIPT'), 'w') as f:
        f.write(script_content)
    _run = uid
    _db_path = db_path


def new_dataset(experiment, kind, **params):
    global _current_dataset, _current_dataset_data
    assert ' ' not in experiment and '-' not in experiment
    assert ' ' not in kind and '-' not in kind

    if not _run:
        _initialize()
    if _current_dataset:
        _write_data(successful=True)

    prefix = experiment + '-' + kind

    if not os.path.exists(os.path.join(_db_path, 'DATA')):
        os.mkdir(os.path.join(_db_path, 'DATA'))

    uid = _make_uid(os.path.join(_db_path, 'DATA'), prefix)
    _current_dataset = os.path.join(_db_path, 'DATA', uid)
    _current_dataset_data = {}

    os.mkdir(_current_dataset)
    yaml.dump({'experiment': experiment,
               'kind': kind,
               'started': datetime.datetime.now(),
               'parameters': params,
               'comment': '',
               'tags': [],
               'protected': False},
               open(os.path.join(_current_dataset, 'INFO'), 'w'))
    os.symlink(os.path.join('..', '..', 'RUNS', _run),
               os.path.join(_current_dataset, 'RUN'))
    os.symlink(os.path.join('..', '..', 'DATA', uid),
               os.path.join(_db_path, 'RUNS', _run, uid))


def add_values(**new_data):
    _check_dataset_keys(new_data.keys())
    _current_dataset_data.update(new_data)


def append_values(**new_data):
    if not _current_dataset:
        raise ValueError('no data set created')
    data = _current_dataset_data
    for k, v in new_data.iteritems():
        if k not in data:
            data[k] = [v]
        elif not isinstance(data[k], list):
            data[k] = [data[k], v]
        else:
            data[k].append(v)


def add_data(**new_data):
    _check_dataset_keys(new_data.keys())

    def dump_file(k, v):
        filename = 'DATA.' + k + '.0'
        dump(v, open(os.path.join(_current_dataset, filename), 'w'))
        return filename

    new_data = {k: DataLoader(dump_file(k, v)) for k, v in new_data.iteritems()}
    add_values(**new_data)


def append_data(**new_data):

    if not _current_dataset:
        raise ValueError('no data set created')

    data = _current_dataset_data

    def dump_file(k, v):
        if k not in data:
            count = 0
        elif not isinstance(data[k], list):
            count = 1
        else:
            count = len(data[k])
        with open(filename, 'w'):
            dump(v, filename)
        return filename

    new_data = {k: DataLoader(dump_file(k, v)) for k, v in new_data.iteritems()}
    append_values(**new_data)


def _write_data(successful):
    assert _current_dataset
    yaml.dump(_current_dataset_data,
              open(os.path.join(_current_dataset, 'DATA'), 'w'),
              width=10000)
    if successful:
        yaml.dump(datetime.datetime.now(),
                  open(os.path.join(_current_dataset, 'FINISHED'), 'w'))


def _check_dataset_keys(new_keys):
    if not _current_dataset:
        raise ValueError('no data set created')
    duplicate_keys = set(_current_dataset_data.keys()).intersection(new_keys)
    if duplicate_keys:
        raise ValueError('Keys {} already exist in data set'.format(duplicate_keys))


def _make_uid(path, prefix=''):
    if prefix:
        prefix = prefix + '-'
    d = datetime.datetime.now()
    while True:
        uid = prefix + datetime.datetime.now().isoformat() + '-' + socket.gethostname()
        if os.path.lexists(os.path.join(path, uid)):
            d = d + datetime.timedelta(microseconds=1)
        else:
            break
    return uid


def _git_info(path):
    from sh import git
    git = git.bake('--no-pager')
    rev_parse = getattr(git, 'rev-parse')
    R = {'branch':    rev_parse('--abbrev-ref', 'HEAD', _cwd=path).strip(),
         'revision':  rev_parse('HEAD', _cwd=path).strip(),
         'untracked': getattr(git, 'ls-files')('--others', '--exclude-standard', _cwd=path).strip(),
         'status':    git.status('-s', _cwd=path).strip(),
         'diff':      git.diff(_cwd=path).strip()}
    R['clean'] = len(R['diff']) == 0
    return R


def _excepthook(exc_type, exc_val, exc_tb):
    global _run
    if _run:
        finished = datetime.datetime.now()
        def dump_failed(filename):
            yaml.dump({'time': finished,
                       'why': repr(exc_val),
                       'traceback': traceback.extract_tb(exc_tb)},
                      open(filename, 'w'))

        if _current_dataset:
            _write_data(successful=False)
            dump_failed(os.path.join(_current_dataset, 'FAILED'))

        dump_failed(os.path.join(_db_path, 'RUNS', _run, 'FAILED'))
        _run = None
    _saved_excepthook(exc_type, exc_val, exc_tb)

_saved_excepthook, sys.excepthook = sys.excepthook, _excepthook


@atexit.register
def _exit_hook():
    if _run:
        finished = datetime.datetime.now()
        if _current_dataset:
            _write_data(successful=True)
            yaml.dump(finished,
                      open(os.path.join(_current_dataset, 'FINISHED'), 'w'))
        yaml.dump(finished,
                  open(os.path.join(_db_path, 'RUNS', _run, 'FINISHED'), 'w'))
