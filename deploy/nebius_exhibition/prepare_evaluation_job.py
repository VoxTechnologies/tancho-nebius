"""Re-run the frozen synthetic final on Nebius AI Cloud against the published v8 revision.

Exhibition run for the Nebius x NVIDIA hackathon. The receipt uses its own schema and
records release_evidence=false, so it cannot be mistaken for Release 1 evidence under
tancho/execution_provider.py, which still requires Modal for training and SITL.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = 'exhibition-eval-v8'
RUN = ROOT / 'runs' / NAME
PACK = ROOT / 'data/observation-program-r1-synthetic-final-v3'
PROJECT = os.environ['NEBIUS_PROJECT_ID']  # e.g. project-xxxxxxxxxxxx
FILESYSTEM = os.environ['NEBIUS_FILESYSTEM_ID']  # compute filesystem holding the parent weights
MODEL_REPO = 'v13s/tancho'
MODEL_REVISION = '1d2b42a3e00542a63bbaf216295c7d119c706b2b'
PARENT = os.environ.get('TANCHO_PARENT_DIR', '/artifacts/inputs/cosmos3-edge-344d602b')
FRAMEWORK_REVISION = '96303bb0bdd14d9efa18f24d8ef98c7a0bfb8412'
CODE = ('training/evaluate_observation_program_export.py',
        'training/observation_program_loader.py', 'training/score_observation_program.py',
        'training/observation_program_legacy_authority.py',
        'training/observation-program-r1-synthetic-final-v3-thresholds.json',
        'tancho/edge_chat.py', 'tancho/contracts.py', 'tancho/mission_context.py',
        'tancho/mission_profiles.py', 'tancho/observation_intent.py',
        'tancho/observation_program.py', 'tancho/__init__.py')

BOOTSTRAP = r'''import hashlib,json,os,subprocess,tarfile,time,traceback
from pathlib import Path
stamp=__STAMP__
remote=Path('/artifacts/exhibition-eval-v8-'+stamp);remote.mkdir(parents=True,exist_ok=False)
scratch=Path('/tmp/tancho-exhibition');scratch.mkdir();code=1
summary={'schema_version':'tancho-exhibition-eval-summary-1.0','stamp':stamp}
try:
 work=remote/'work';work.mkdir()
 archive=Path('/opt/tancho/code.tar.gz')
 with archive.open('rb') as f:
  assert hashlib.file_digest(f,'sha256').hexdigest()==__CODE_SHA__,'code archive changed'
 with tarfile.open(archive) as tf:tf.extractall(work,filter='data')
 pack=None
 for base,dirs,files in os.walk('/artifacts'):
  if 'manifest.json' in files and base.endswith('data/observation-program-r1-synthetic-final-v3'):
   with (Path(base)/'manifest.json').open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()==__PACK_SHA__:pack=Path(base);break
 if pack is None:raise FileNotFoundError('frozen pack not found on /artifacts')
 summary['pack']=str(pack)
 parent=Path(__PARENT__)
 if not parent.is_dir():raise FileNotFoundError(parent)
 model=scratch/'published'
 subprocess.run(['uvx','--from','huggingface_hub[cli]','hf','download',__REPO__,
                 '--revision',__REV__,'--local-dir',str(model),'--quiet'],check=True)
 subprocess.run(['python3',str(model/'verify_files.py')],cwd=model,check=True)
 framework=scratch/'cosmos-framework'
 subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                 'https://github.com/NVIDIA/cosmos-framework.git',str(framework)],check=True)
 subprocess.run(['git','-C',str(framework),'fetch','--depth','1','origin',__FRAMEWORK__],check=True)
 subprocess.run(['git','-C',str(framework),'checkout','--detach',__FRAMEWORK__],check=True)
 subprocess.run(['uv','sync','--frozen','--all-extras','--group=cu130-train','--no-cache'],
                cwd=framework,check=True)
 python=framework/'.venv/bin/python'
 env=dict(os.environ);env['PATH']=str(framework/'.venv/bin')+':'+env['PATH']
 env['PYTHONPATH']=str(work);env['LD_LIBRARY_PATH']='';env['HF_HUB_OFFLINE']='1'
 env['TRANSFORMERS_OFFLINE']='1';env['WANDB_MODE']='disabled'
 command=[str(python),str(work/'training/evaluate_observation_program_export.py'),
  '--pack',str(pack),'--base',str(model),'--export',str(model),
  '--thresholds',str(work/'training/observation-program-r1-synthetic-final-v3-thresholds.json'),
  '--output',str(remote/'results')]
 started=time.time()
 with (remote/'controller.log').open('w') as log:
  code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=4200).returncode
 summary['seconds']=round(time.time()-started,1);summary['exit_code']=code
 for name in ('summary.json','scores.json','result.json'):
  path=remote/'results'/name
  if path.is_file():
   summary['results_'+name]=json.loads(path.read_text());break
 summary['results_files']=sorted(p.name for p in (remote/'results').glob('*')) if (remote/'results').is_dir() else []
except BaseException:
 with (remote/'bootstrap-error.log').open('w') as log:traceback.print_exc(file=log)
 summary['error']=traceback.format_exc()[-600:]
(remote/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
print('TANCHO_EVAL '+json.dumps(summary,separators=(',',':'))[:3000],flush=True)
raise SystemExit(code)
'''


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    code_archive = RUN / 'code.tar.gz'
    with tarfile.open(code_archive, 'w:gz', compresslevel=9) as output:
        for name in sorted(CODE):
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            output.addfile(info, io.BytesIO(data))
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    bootstrap = (BOOTSTRAP
                 .replace('__STAMP__', repr(stamp))
                 .replace('__CODE_SHA__', repr(digest(code_archive)))
                 .replace('__PACK_SHA__', repr(digest(PACK / 'manifest.json')))
                 .replace('__PARENT__', repr(PARENT))
                 .replace('__REPO__', repr(MODEL_REPO))
                 .replace('__REV__', repr(MODEL_REVISION))
                 .replace('__FRAMEWORK__', repr(FRAMEWORK_REVISION)))
    compile(bootstrap, 'bootstrap', 'exec')
    (RUN / 'bootstrap.py').write_text(bootstrap)
    with (RUN / 'bootstrap.py.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as stream:
            stream.write(bootstrap.encode())
    (RUN / 'launcher.py').write_text(
        "import gzip\nexec(compile(gzip.open('/opt/tancho/bootstrap.py.gz').read(),"
        "'bootstrap.py','exec'))\n")
    injected = sum((RUN / n).stat().st_size for n in ('launcher.py', 'bootstrap.py.gz', 'code.tar.gz'))
    if injected >= 48 * 1024:
        raise ValueError('Injected payload is %d bytes, over the 48 KiB limit' % injected)
    image = json.loads((ROOT / 'runs/training-preparation/image-push/push-state.json').read_text())
    key = RUN / 'ssh_key'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', NAME,
                        '-f', str(key)], check=True)
    key.chmod(0o600)
    command = [str(Path.home() / '.local/bin/nebius'), 'ai', 'job', 'create',
               '--parent-id', PROJECT, '--name', 'tancho-' + NAME,
               '--platform', 'gpu-h100-sxm', '--preset', '1gpu-16vcpu-200gb',
               '--image', image['image_ref'],
               '--container-command', 'python', '--args', '/opt/tancho/launcher.py',
               '--volume', FILESYSTEM + ':/artifacts:rw',
               '--ssh-key', (RUN / 'ssh_key.pub').read_text().strip(), '--timeout', '90m',
               '--restart-policy', 'never', '--disk-size', '200Gi', '--shm-size', '16Gi',
               '--inject-file', str(RUN / 'launcher.py') + ':/opt/tancho/launcher.py',
               '--inject-file', str(RUN / 'bootstrap.py.gz') + ':/opt/tancho/bootstrap.py.gz',
               '--inject-file', str(code_archive) + ':/opt/tancho/code.tar.gz',
               '--no-browser', '--no-check-update', '--no-progress',
               '--auth-timeout', '30s', '--per-retry-timeout', '30s',
               '--format', 'json', '--retries', '1',
               '--async' if args.execute else '--dry-run']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    stem = 'create' if args.execute else 'dry-run'
    (RUN / (stem + '.stdout')).write_text(completed.stdout)
    (RUN / (stem + '.stderr')).write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    (RUN / 'receipt.json').write_text(json.dumps({
        'schema_version': 'tancho-exhibition-receipt-1.0', 'purpose': 'exhibition',
        'release_evidence': False, 'provider': 'nebius', 'compute': 'gpu-h100-sxm 1gpu-16vcpu-200gb',
        'job_name': 'tancho-' + NAME, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
        'framework_revision': FRAMEWORK_REVISION, 'image_ref': image['image_ref'],
        'injected_bytes': injected, 'stamp': stamp, 'executed': args.execute}, indent=2) + '\n')
    print(completed.stdout.strip())
    print('injected bytes:', injected)


if __name__ == '__main__':
    main()
