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

from cPickle import load

import functools
import glob
import itertools
import os
import shutil
import textwrap
import yaml

import numpy as np


class DataLoader(object):

    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        return load(open(self.filename))


def data_loader_representer(dumper, data):
    return dumper.represent_scalar(u'!DataLoader', data.filename)


def data_loader_constructor(loader, node, path=None):
    filename = loader.construct_scalar(node)
    if path is not None:
        filename = os.path.join(path, filename)
    return DataLoader(filename)


yaml.add_representer(DataLoader, data_loader_representer)


class Dataset(object):

    class DataDict(object):

        def __getitem__(self, name):
            return getattr(self, name)

        def __setitem__(self, name, value):
            setattr(self, name, value)

        def __iter__(self):
            return iter(self.__dict__)

        @property
        def dict(self):
            return self.__dict__

        def __repr__(self):
            return ', '.join(str(k) + ': ' + str(v) for k, v in sorted(self.dict.iteritems()))


    def __init__(self, path):
        self.path = path

        info = yaml.load(open(os.path.join(path, 'INFO')))
        self._params = info.pop('parameters')
        self.p = self.DataDict()
        self.p.__dict__ = self._params
        self.tags = frozenset(info.pop('tags'))
        self.__dict__.update(info)

        self.failed = False
        if os.path.exists(os.path.join(path, 'FINISHED')):
            self.finished = yaml.load(open(os.path.join(path, 'FINISHED')))
        else:
            self.finished = False
            if os.path.exists(os.path.join(path, 'FAILED')):
                self.failed = yaml.load(open(os.path.join(path, 'FAILED')))

        self._locked = True
        self._deleted = False

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def d(self):
        if self._deleted:
            raise ValueError('Dataset has been deleted')
        if hasattr(self, '_d'):
            return self._d

        class MyLoader(yaml.CLoader):
            pass

        MyLoader.add_constructor(u'!DataLoader',
                                 functools.partial(data_loader_constructor, path=self.path))
        self._data = yaml.load(open(os.path.join(self.path, 'DATA')), Loader=MyLoader)

        def try_array(x):
            if isinstance(x, list):
                a = np.array(x)
                if a.dtype == np.object:
                    return x
                else:
                    return a
            else:
                return x
        self._data = {k: try_array(v) for k, v in self._data.iteritems()}

        self._d = self.DataDict()
        self._d.__dict__ = self._data
        return self._d

    def delete(self):
        if self._deleted:
            return
        shutil.rmtree(self.path)
        self._deleted = True

    def tag(self, *args):
        if self._deleted:
            raise ValueError('Dataset has been deleted')
        self.__dict__['tags'] = frozenset(self.tags.union(args))
        self._update_info()

    def untag(self, *args):
        if self._deleted:
            raise ValueError('Dataset has been deleted')
        self.__dict__['tags'] = frozenset(self.tags - set(args))
        self._update_info()

    def _update_info(self):
        data = {k: getattr(self, k) for k in ['experiment', 'kind', 'started',
                                              'comment', 'protected']}
        data['parameters'] = self.p.dict
        data['tags'] = list(sorted(self.tags))

        yaml.dump(data,
                  open(os.path.join(self.path, 'INFO'), 'w'))

    def __setattr__(self, k, v):
        if not hasattr(self, '_locked'):
            super(Dataset,self).__setattr__(k, v)
        else:
            if getattr(self, '_deleted', False):
                raise ValueError('Dataset has been deleted')
            if k not in ['comment', 'protected'] and not k.startswith('_'):
                raise AttributeError('Cannot change attribute ' + k)
            if k == 'protected' and not isinstance(v, bool):
                raise AttributeError('protected must be bool')
            if k == 'comment' and not isinstance(v, str):
                raise AttributeError('comment must be string')
            super(Dataset, self).__setattr__(k, v)
            if not k.startswith('_'):
                self._update_info()

    def __str__(self):
        params = '\n'.join(textwrap.wrap(', '.join('{}={}'.format(k, v) for k, v in sorted(self.p.dict.iteritems())),
                                          initial_indent=' '*10, subsequent_indent=' '*10, width=100))
        s = '{} {}'.format(self.name,
                            '✓' if self.finished else '✗' if self.failed else '?')
        if self._deleted:
            s += ' ***DELETED***'
        s += '\n' + params
        return s

    def __repr__(self):
        return "Dataset('{}')".format(self.path)


class DatasetCollection(object):

    def __init__(self, datasets):
        self.datasets = datasets

    def dir(self):
        print(str(self))

    def select(self, *args, **kwargs):
        def selector(ds):
            try:
                for k, v in kwargs.iteritems():
                    if ds.p.dict[k] != v:
                        return False
                return all(f(ds) for f in args)
            except:
                return False
        return DatasetCollection(filter(selector, self.datasets))

    def select_unique(self, *args, **kwargs):
        ds = self.select(*args, **kwargs)
        if len(ds) == 0:
            raise ValueError('No matching datasets found!')
        elif len(ds) > 1:
            raise ValueError('More than one ({}) datasetes found!'.format(len(ds)))
        else:
            return ds[0]

    def select_last(self, *args, **kwargs):
        ds = self.select(*args, **kwargs)
        if len(ds) == 0:
            raise ValueError('No matching datasets found!')
        else:
            return ds[-1]

    def delete(self):
        for ds in self.datasets:
            ds.delete()

    def __getitem__(self, n):
        return self.datasets[n]

    def __len__(self):
        return len(self.datasets)

    def __repr__(self):
        return 'DatasetCollection([' + ',\n'.join(repr(ds) for ds in self.datasets) + '])'

    def __str__(self):
        if not self.datasets:
            return 'None!'
        return '\n\n'.join(map(str, self.datasets))


class SimulationDatabase(object):

    def __init__(self, db_path=None):
        if not db_path:
            db_path = os.environ['SIMDB_PATH']
        self.db_path = db_path
        combinations = sorted(set([tuple(s.split('-')[:2]) for s in os.listdir(os.path.join(db_path, 'DATA'))]))
        self.experiments = {k: [v2[1] for v2 in v] for k, v in itertools.groupby(combinations, lambda x:x[0])}

    def select(self, pattern, *args, **kwargs):
        pattern = '*' + pattern + '*'
        paths = sorted(glob.glob(os.path.join(self.db_path, 'DATA', pattern)))
        ds = DatasetCollection(map(Dataset, paths))
        if args or kwargs:
            return ds.select(*args, **kwargs)
        else:
            return ds

    def select_unique(self, pattern, *args, **kwargs):
        ds = self.select(pattern, *args, **kwargs)
        if len(ds) == 0:
            raise ValueError('No matching datasets found!')
        elif len(ds) > 1:
            raise ValueError('More than one ({}) datasetes found!'.format(len(ds)))
        else:
            return ds[0]

    def select_last(self, pattern, *args, **kwargs):
        ds = self.select(pattern, *args, **kwargs)
        if len(ds) == 0:
            raise ValueError('No matching datasets found!')
        else:
            return ds[-1]

    def delete(self, pattern, *args, **kwargs):
        ds = self.select(pattern, *args, **kwargs)
        print('Deleting:')
        ds.dir()
        ds.delete()

    def dir(self):
        print(str(self))

    def __str__(self):
        maxlen = max(map(len, self.experiments.keys())) + 1
        out = ''
        for k, v in self.experiments.iteritems():
            out += ('{:{maxlen}} {}\n'.format(k + ':', v, maxlen=maxlen))
        return out[:-1]

    def __repr__(self):
        return "SimulationDatabase('{}')".format(self.db_path)
