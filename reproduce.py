#!/usr/bin/env python3
"""Inspect and run the curated ISBC experiments from this repository."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, shlex, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def read_json(path):
    return json.loads((ROOT/path).read_text())

def training_command(name, mode='train', device=None, precision=None):
    entries={r['name']:r for r in read_json('configs/training/index.json')}
    if name not in entries: raise ValueError(f'Unknown training configuration: {name}')
    config=read_json(entries[name]['config'])
    # Read argument declarations without importing Torch or loading a backbone.
    tree=ast.parse((ROOT/'binary_factors_train.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
    options={}
    for call in ast.walk(fn):
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument'):continue
        flags=[a.value for a in call.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')]
        if not flags:continue
        kw={x.arg:x.value.value for x in call.keywords if isinstance(x.value,ast.Constant)}
        dest=kw.get('dest',flags[0][2:].replace('-','_'))
        options.setdefault(dest,[]).append((flags[0],kw.get('action','store')))
    config['mode']=mode
    if device:config['device']=device
    if precision:config['precision']=precision
    # A new environment must be allowed to extract embeddings if no cache exists.
    config['require_cached_embeddings']=False
    config['resume']=False
    command=[sys.executable,'binary_factors_train.py']
    for key,variants in options.items():
        if key not in config:continue
        value=config[key]
        if isinstance(value,bool):
            match=next((flag for flag,action in variants if (action=='store_true' and value) or (action=='store_false' and not value)),None)
            if match:command.append(match)
        elif value is not None:command.extend([variants[0][0],str(value)])
    return command,config

def restore_splits():
    items=read_json('reproduction/splits/index.json')
    # Preflight every destination before writing any file.
    for r in items:
        src=ROOT/r['file'];dest=ROOT/r['source']
        if hashlib.sha256(src.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(f'Corrupted split: {src}')
        if dest.exists() and dest.read_bytes()!=src.read_bytes():raise ValueError(f'Refusing to replace different split: {dest}')
    for r in items:
        src=ROOT/r['file'];dest=ROOT/r['source'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(src,dest)
    print(f'Restored {len(items)} recorded ID splits.')

def execute(command):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'isbc-matplotlib'))
    print(shlex.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)

def verify():
    issues=[];count=0
    for p in ROOT.rglob('*.py'):
        if any(x in p.parts for x in ['.venv','__pycache__','.external_methods']):continue
        try:ast.parse(p.read_text(),filename=str(p));count+=1
        except SyntaxError as e:issues.append(str(e))
    for entry in read_json('configs/training/index.json'):
        _,cfg=training_command(entry['name'])
        if cfg['objective']!='sparse_ae':issues.append(entry['name']+': not sparse_ae')
        if cfg['target_active']>cfg['k_factors']:issues.append(entry['name']+': invalid activity')
    for name,flow in read_json('reproduction/workflows.json').items():
        for c in flow['commands']:
            if c[0] in ['python','Rscript','bash'] and not (ROOT/c[1]).is_file():issues.append(f'{name}: missing {c[1]}')
    print(f'Parsed {count} Python files; checked configuration and workflow paths.')
    if issues:raise ValueError('\n'.join(issues))
    print('Static checks passed. This does not execute experiments or validate their numerical results.')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('list');sub.add_parser('verify');sub.add_parser('restore-splits')
    for name in ['train','encode']:
        s=sub.add_parser(name);s.add_argument('configuration');s.add_argument('--device',choices=['auto','cpu','mps','cuda']);s.add_argument('--precision',choices=['auto','fp32','fp16','bf16']);s.add_argument('--execute',action='store_true',help='Run the displayed command; otherwise only print it.')
    s=sub.add_parser('run');s.add_argument('workflow');s.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.action=='list':
        print('TRAINING CONFIGURATIONS')
        for r in read_json('configs/training/index.json'):print(r['name'])
        print('\nANALYSIS WORKFLOWS')
        for name,flow in read_json('reproduction/workflows.json').items():print(name+': '+flow['description'])
    elif args.action=='verify':verify()
    elif args.action=='restore-splits':restore_splits()
    elif args.action in ['train','encode']:
        command,cfg=training_command(args.configuration,args.action,args.device,args.precision)
        if args.execute:
            if not (ROOT/cfg['input']).is_file():raise ValueError('Prepare the input dataset first: '+cfg['input'])
            if args.action=='train' and (ROOT/cfg['output_dir']/'best/model.pt').exists():raise ValueError('A fitted checkpoint already exists. Use the trainer directly if you intentionally want to resume or replace a run.')
            execute(command)
        else:print(shlex.join(command))
    elif args.action=='run':
        flows=read_json('reproduction/workflows.json')
        if args.workflow not in flows:raise ValueError('Unknown workflow: '+args.workflow)
        flow=flows[args.workflow];print(flow['description']);print('Prerequisites: '+flow['prerequisites'])
        for command in flow['commands']:
            command=[sys.executable if x=='python' and i==0 else x for i,x in enumerate(command)]
            if args.execute:execute(command)
            else:print(shlex.join(command))
if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as exc:sys.exit(str(exc))
