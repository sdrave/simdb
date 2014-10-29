#!/usr/bin/env python
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

from __future__ import print_function

import os
import sys

import yaml



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

yaml.add_representer(unicode, unicode_representer)
yaml.add_representer(str, str_representer)


path = sys.argv[1]


for ds in os.listdir(os.path.join(path, 'DATA')):
    info = yaml.load(open(os.path.join(path, 'DATA', ds, 'INFO')))
    info['experiment'] = info['experiment'] + '_' + info.pop('kind')
    yaml.dump(info, open(os.path.join(path, 'DATA', ds, 'INFO'), 'w'))
    os.rename(os.path.join(path, 'DATA', ds),
              os.path.join(path, 'DATA', '-'.join((lambda l: [l[0] + '_' + l[1]] + l[2:])(ds.split('-')))))

for run in os.listdir(os.path.join(path, 'RUNS')):
    for ds in os.listdir(os.path.join(path, 'RUNS', run)):
        if ds in ('INFO', 'SCRIPT', 'FINISHED', 'FAILED'):
            continue
        new_name = '-'.join((lambda l: [l[0] + '_' + l[1]] + l[2:])(ds.split('-')))
        os.unlink(os.path.join(path, 'RUNS', run, ds))
        os.symlink(os.path.join('..', '..', 'DATA', new_name),
                   os.path.join(path, 'RUNS', run, new_name))
