# -*- coding: utf-8 -*-
# Copyright (c) 2014, 2015, Stephan Rave
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

from cPickle import dumps, load

import fnmatch
import itertools
import logging
import os
import shutil
import textwrap

import numpy as np
import yaml

logger = logging.getLogger('simdb')


class DataLoader(object):

    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        return load(open(self.filename))


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

        if not self.finished:
            if self.failed:
                logger.warn('Loading data of failed dataset {}.'.format(self.name))
            else:
                if os.path.exists(os.path.join(self.path, 'DATA')):
                    logger.warn('Loading data of unfinished dataset {}.'.format(self.name))
                else:
                    raise ValueError('No data has been written to unfinished dataset {}.'.format(self.name))

        self._data = load(open(os.path.join(self.path, 'DATA')))
        for v in self._data.itervalues():
            if isinstance(v, DataLoader):
                v.filename = os.path.join(self.path, v.filename)
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, DataLoader):
                        vv.filename = os.path.join(self.path, vv.filename)

        self._d = self.DataDict()
        self._d.__dict__ = self._data
        return self._d

    @property
    def t(self):
        if self._deleted:
            raise ValueError('Dataset has been deleted')
        if hasattr(self, '_t'):
            return self._t

        if not self.finished:
            if self.failed:
                logger.warn('Loading data of failed dataset {}.'.format(self.name))
            else:
                if os.path.exists(os.path.join(self.path, 'TIMES')):
                    logger.warn('Loading data of unfinished dataset {}.'.format(self.name))
                else:
                    raise ValueError('No data has been written to unfinished dataset {}.'.format(self.name))

        self._times = {k: np.array(v)
                       for k, v in yaml.load(open(os.path.join(self.path, 'TIMES')))['duration'].iteritems()}

        self._t = self.DataDict()
        self._t.__dict__ = self._times
        return self._t

    @property
    def host(self):
        return self.name.split('-')[-1]

    def unload_data(self):
        if hasattr(self, '_d'):
            del self._d
            del self._data
        if hasattr(self, '_t'):
            del self._t
            del self._times

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
        data = {k: getattr(self, k) for k in ['experiment', 'started', 'comment', 'protected']}
        data['parameters'] = self.p.dict
        data['tags'] = list(sorted(self.tags))

        yaml.dump(data,
                  open(os.path.join(self.path, 'INFO'), 'w'))

    def __setattr__(self, k, v):
        if not hasattr(self, '_locked'):
            super(Dataset, self).__setattr__(k, v)
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
        failed = kwargs.pop('failed', None)

        def selector(ds):
            if failed is not None and bool(ds.failed) != failed:
                return False
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

    def duplicates(self, params=None):
        if params:
            params = sorted(params)
            keyfunc = lambda ds: (ds.experiment,
                                  list(dumps(ds.p[k], protocol=-1) for k in params))
            groups = itertools.groupby(sorted((ds for ds in self.datasets if set(ds.p.dict.keys()) >= set(params)),
                                              key=keyfunc),
                                       key=keyfunc)
        else:
            keyfunc = lambda ds: (ds.experiment,
                                  list(sorted((k, dumps(v, protocol=-1)) for k, v in ds.p.dict.iteritems())))
            groups = itertools.groupby(sorted(self.datasets, key=keyfunc), key=keyfunc)

        def get_duplicates(groups):
            for k, v in groups:
                datasets = list(v)
                if len(datasets) > 1:
                    p = datasets[0].p.dict
                    info = {'experiment': datasets[0].experiment,
                            'params': p if not params else {k: p[k] for k in params}}
                    yield [info, datasets]

        class DuplicatesList(list):
            def __str__(self):
                formatter = lambda i, ds: ('experiment: ' + i['experiment'] + '\n' +
                                           'params:     ' +
                                           '\n'.join(textwrap.wrap(', '.join('{}={}'.format(k, v)
                                                                   for k, v in sorted(i['params'].iteritems())),
                                                                   initial_indent='',
                                                                   subsequent_indent=' ' * len('experiment: '),
                                                                   width=100)) + '\n' +
                                           'count:      ' + str(len(ds)) + '\n')
                return '\n'.join(formatter(i, ds) for i, ds in self)

            def delete_old(self):
                for i, datasets in self:
                    for ds in datasets[:-1]:
                        ds.delete()

            __repr__ = __str__

        return DuplicatesList(get_duplicates(groups))

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
        self.experiments = sorted(set(s.split('-')[0] for s in os.listdir(os.path.join(db_path, 'DATA'))))

    def select(self, pattern, *args, **kwargs):
        paths = sorted(os.path.join(self.db_path, 'DATA', fn) for fn in os.listdir(os.path.join(self.db_path, 'DATA'))
                       if fnmatch.fnmatch(fn.split('-')[0], pattern))
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

    def duplicates(self, params=None):
        return self.select('*').duplicates(params)

    def dir(self):
        print(str(self))

    def __str__(self):
        return str(self.experiments)

    def __repr__(self):
        return "SimulationDatabase('{}')".format(self.db_path)
