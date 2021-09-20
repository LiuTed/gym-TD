import os
import shutil
import re
from collections import OrderedDict
import subprocess
import numpy as np

class Result:
    checkpoint = None
    log = None
    tarball = None
    board = None

if __name__ == '__main__':
    results = OrderedDict()

    def load_files():
        files = os.listdir()
        for f in files:
            res = re.match('ckpt-([0-9]{6}-[0-9]{6})', f)
            if res is not None:
                val = results.get(res.group(1), Result())
                val.checkpoint = f
                results[res.group(1)] = val
            res = re.match('Result-([0-9]{6}-[0-9]{6})', f)
            if res is not None:
                val = results.get(res.group(1), Result())
                val.log = f
                results[res.group(1)] = val
            res = re.match('result-([0-9]{6}-[0-9]{6}).tar.gz', f)
            if res is not None:
                val = results.get(res.group(1), Result())
                val.tarball = f
                results[res.group(1)] = val
    
    def result_list():
        for i, (d, r) in enumerate(results.items()):
            print(
                '{}: {} {} {} {} {}'.format(
                    i,
                    d,
                    'T' if r.checkpoint is not None else 'F',
                    'T' if r.log is not None else 'F',
                    'T' if r.tarball is not None else 'F',
                    'R' if r.board is not None else '-'
                )
            )
    
    def get_key_val(sid, keys, show=True):
        if len(sid.split('-')) == 1:
            id = int(sid)
            if id >= len(results):
                if show:
                    print('Unknown index', sid)
                return None, None
            key = keys[id]
        else:
            key = sid
        try:
            val = results[key]
        except:
            if show:
                print('Unknown time', key)
            return None, None
        return key, val
        
    print('Checking files')
    load_files()

    print('Results:')
    result_list()

    while True:
        cmd = input('> ')
        values = list(results.values())
        keys = list(results.keys())

        if cmd.strip() == '':
            continue

        if cmd == 'ls' or cmd == 'list':
            result_list()
            continue
        
        if cmd == 'exit':
            break

        res = re.match('rm( checkpoint| log| tarball)*( [0-9]+| [0-9]{6}-[0-9]{6})+\s*$', cmd)
        if res is not None:
            actions = 0
            dkeys = []
            for sid in res.groups():
                if sid is None:
                    continue
                if sid == ' checkpoint':
                    actions |= 1
                    continue
                if sid == ' log':
                    actions |= 2
                    continue
                if sid == ' tarball':
                    actions |= 4
                    continue
                key, val = get_key_val(sid, keys)
                if key is None:
                    continue
                dkeys.append(key)
            if actions == 0:
                actions = 7
            dkeys = np.unique(dkeys)
            if len(dkeys) == 0:
                print('Has nothing to delete')
                continue
            print('Deleting the{}{}{} of the following results:'.format(
                ' checkpoint' if actions & 1 != 0 else '',
                ' log' if actions & 2 != 0 else '',
                ' tarball' if actions & 4 != 0 else ''
            ))
            for key in dkeys:
                print(key)
            ck = input('[y/N] ')
            if ck.upper() == 'Y':
                for key in dkeys:
                    val = results[key]
                    if actions & 1 != 0 and val.checkpoint is not None:
                        print('Deleting', val.checkpoint)
                        shutil.rmtree(val.checkpoint)
                        val.checkpoint = None
                    if actions & 2 != 0 and val.log is not None:
                        if val.board is not None:
                            print('Closing the tensorboard process of result {}'.format(key))
                            val.board.terminate()
                            val.board.wait(10)
                            if val.board.poll() is None:
                                val.board.kill()
                            val.board = None
                        print('Deleting', val.log)
                        shutil.rmtree(val.log)
                        val.log = None
                    if actions & 4 != 0 and val.tarball is not None:
                        print('Deleting', val.tarball)
                        os.remove(val.tarball)
                        val.tarball = None
                    if val.checkpoint or val.log or val.tarball:
                        results[key] = val
                    else:
                        results.pop(key)
            load_files()
            continue

        res = re.match('board ([0-9]+|[0-9]{6}-[0-9]{6})\s*$', cmd)
        if res is not None:
            sid = res.group(1)
            key, val = get_key_val(sid, keys)
            if key is None:
                continue
            if val.board is not None:
                print('board of {} is running'.format(key))
                continue
            if val.log is None:
                print('log of {} does not exists'.format(key))
                continue
            subp = subprocess.Popen(
                'tensorboard --logdir={} --bind_all'.format(val.log),
                shell = True,
                stdout = subprocess.DEVNULL
            )
            val.board = subp
            results[key] = val
            continue

        res = re.match('stop ([0-9]+|[0-9]{6}-[0-9]{6})\s*$', cmd)
        if res is not None:
            sid = res.group(1)
            key, val = get_key_val(sid, keys)
            if key is None:
                continue
            if val.board is None:
                print('board of {} is not running'.format(key))
                continue
            val.board.terminate()
            val.board.wait()
            val.board = None
            results[key] = val
            continue

        res = re.match('pack ([0-9]+|[0-9]{6}-[0-9]{6})\s*$', cmd)
        if res is not None:
            sid = res.group(1)
            key, val = get_key_val(sid, keys)
            if key is None:
                continue
            if val.tarball is not None:
                print('tarball of {} has already existed'.format(key))
                continue
            subp = subprocess.Popen(
                'tar czvf result-{}.tar.gz {} {}'.format(
                    key, val.checkpoint or '', val.log or ''
                ),
                shell = True
            )
            subp.wait()
            val.tarball = 'result-{}.tar.gz'.format(key)
            results[key] = val
            continue

        res = re.match('unpack ([0-9]+|[0-9]{6}-[0-9]{6})\s*$', cmd)
        if res is not None:
            sid = res.group(1)
            key, val = get_key_val(sid, keys)
            if key is None:
                continue
            if val.tarball is None:
                print('tarball of {} does not exist'.format(key))
                continue
            subp = subprocess.Popen(
                'tar xzvf {}'.format(val.tarball),
                shell = True
            )
            subp.wait()
            load_files()
            continue
        
        if cmd != 'help':
            print('Unknown command', cmd)
        print('''Usage:
    help: show this message
    ls: list the status of results with format \'ID: time has_checkpoint has_log has_tarball if_tensorboard_running\'
    rm [checkpoint] [log] [tarball] id/time[ id/time[ ...]]: remove the results listed (double check needed)
    board id/time: execute tensorboard to visualize the result specified
    stop id/time: stop tensorboard of that result
    pack id/time: pack the result into tar ball
    unpack id/time: unpack the tar ball of result
    exit: exit'''
        )
            
        
