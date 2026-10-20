"""List the artifact filesystem from a bounded CPU job, to see what a Nebius rerun can still use.

Exhibition preflight for the Nebius x NVIDIA hackathon. This is not Release 1 evidence:
nothing here is scored, and the receipt carries its own schema so it cannot be mistaken
for a provider receipt under tancho/execution_provider.py.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAME = 'exhibition-artifact-inventory-v2'
RUN = ROOT / 'runs' / NAME
PROJECT = os.environ['NEBIUS_PROJECT_ID']  # e.g. project-xxxxxxxxxxxx
FILESYSTEM = os.environ['NEBIUS_FILESYSTEM_ID']  # compute filesystem holding the parent weights
IMAGE = ('docker.io/library/python@sha256:'
         '44ff437bba879d4941b710a369a8f19266aea34b29002807f0c487fabc9eec9b')

BOOTSTRAP = r'''import json,os,traceback
from pathlib import Path
root=Path('/artifacts')
def entry(path,depth):
 row={'name':path.name,'dir':path.is_dir()}
 if path.is_file():row['bytes']=path.stat().st_size
 if path.is_dir() and depth>0:
  try:row['children']=[entry(c,depth-1) for c in sorted(path.iterdir())[:60]]
  except PermissionError:row['children']='permission denied'
 return row
try:
 tree=entry(root,3)
 probes={}
 for name in ('inputs','observation-program-r1-training-v4','observation-program-r1-synthetic-final-v3'):
  probes[name]=(root/name).is_dir()
 inputs=root/'inputs'
 probes['inputs_children']=[p.name for p in sorted(inputs.iterdir())] if inputs.is_dir() else []
 total=0
 for base,_,files in os.walk(root):
  for f in files:
   try:total+=(Path(base)/f).stat().st_size
   except OSError:pass
 out={'schema_version':'tancho-exhibition-inventory-1.0','total_bytes':total,'probes':probes,
      'top':[c['name'] for c in tree.get('children',[])]}
 print('TANCHO_INVENTORY '+json.dumps(out,separators=(',',':')),flush=True)
except BaseException:
 traceback.print_exc()
 raise SystemExit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    compile(BOOTSTRAP, 'bootstrap', 'exec')
    (RUN / 'bootstrap.py').write_text(BOOTSTRAP)
    with (RUN / 'bootstrap.py.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as stream:
            stream.write(BOOTSTRAP.encode())
    (RUN / 'launcher.py').write_text(
        "import gzip\nexec(compile(gzip.open('/opt/tancho/bootstrap.py.gz').read(),"
        "'bootstrap.py','exec'))\n")
    key = RUN / 'ssh_key'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', NAME,
                        '-f', str(key)], check=True)
    key.chmod(0o600)
    command = [str(Path.home() / '.local/bin/nebius'), 'ai', 'job', 'create',
               '--parent-id', PROJECT, '--name', 'tancho-' + NAME,
               '--platform', 'cpu-d3', '--preset', '4vcpu-16gb', '--image', IMAGE,
               '--container-command', 'python', '--args', '/opt/tancho/launcher.py',
               '--volume', FILESYSTEM + ':/artifacts:ro',
               '--ssh-key', (RUN / 'ssh_key.pub').read_text().strip(), '--timeout', '20m',
               '--restart-policy', 'never', '--disk-size', '32Gi', '--shm-size', '1Gi',
               '--inject-file', str(RUN / 'launcher.py') + ':/opt/tancho/launcher.py',
               '--inject-file', str(RUN / 'bootstrap.py.gz') + ':/opt/tancho/bootstrap.py.gz',
               '--no-browser', '--no-check-update', '--no-progress',
               '--auth-timeout', '30s', '--per-retry-timeout', '30s',
               '--format', 'json', '--retries', '1',
               '--async' if args.execute else '--dry-run']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    stem = 'create' if args.execute else 'dry-run'
    (RUN / (stem + '.stdout')).write_text(completed.stdout)
    (RUN / (stem + '.stderr')).write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    (RUN / 'receipt.json').write_text(json.dumps({
        'schema_version': 'tancho-exhibition-receipt-1.0', 'purpose': 'exhibition',
        'release_evidence': False, 'provider': 'nebius', 'compute': 'cpu-d3 4vcpu-16gb',
        'job_name': 'tancho-' + NAME, 'executed': args.execute}, indent=2) + '\n')
    print(completed.stdout.strip())


if __name__ == '__main__':
    main()
