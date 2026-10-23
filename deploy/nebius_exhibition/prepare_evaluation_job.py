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
NAME = 'exhibition-eval-v8-pack8'
RUN = ROOT / 'runs' / NAME
PUBLIC_REPO = 'https://github.com/VoxTechnologies/tancho-nebius.git'
PUBLIC_COMMIT = '8a40013af9c1fc7d5790c4718e8f80bb3b6283a5'
PACK_MANIFEST_SHA = '27dbeac68d8c5df1127ea8d4d6bf5235d19843852ce2360922e70c18f5f6881f'
PACK_DIR = 'data/observation-program-r1-synthetic-final-v8'
THRESHOLDS = 'training/observation-program-r1-synthetic-final-v8-thresholds.json'
# Your own Nebius resources. The parent weights live on the compute filesystem; the image is
# image is one that carries torch, transformers and git, pushed to a registry the job can read.
PROJECT = os.environ['NEBIUS_PROJECT_ID']
FILESYSTEM = os.environ['NEBIUS_FILESYSTEM_ID']
IMAGE_REF = os.environ['NEBIUS_IMAGE_REF']
MODEL_REPO = 'v13s/tancho'
MODEL_REVISION = '1d2b42a3e00542a63bbaf216295c7d119c706b2b'
PARENT = '/artifacts/inputs/cosmos3-edge-344d602b'
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
 work=scratch/'tancho-nebius'
 subprocess.run(['git','clone','--quiet',__PUBLIC_REPO__,str(work)],check=True)
 subprocess.run(['git','-C',str(work),'checkout','--quiet','--detach',__PUBLIC_COMMIT__],check=True)
 pack=work/__PACK_DIR__
 with (pack/'manifest.json').open('rb') as f:
  assert hashlib.file_digest(f,'sha256').hexdigest()==__PACK_SHA__,'frozen pack changed'
 summary['pack']=__PACK_DIR__;summary['public_commit']=__PUBLIC_COMMIT__
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
  '--thresholds',str(work/__THRESHOLDS__),
  '--output',str(remote/'results')]
 started=time.time()
 with (remote/'controller.log').open('w') as log:
  code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=4200).returncode
 summary['seconds']=round(time.time()-started,1);summary['exit_code']=code
 for name in ('report.json',):
  path=remote/'results'/name
  if path.is_file():
   r=json.loads(path.read_text());summary['passed']=r.get('passed')
   summary['score']=r.get('score',{}).get('overall');break
 summary['results_count']=len(list((remote/'results').glob('*.txt'))) if (remote/'results').is_dir() else 0
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
                 .replace('__PACK_SHA__', repr(PACK_MANIFEST_SHA))
                 .replace('__PUBLIC_REPO__', repr(PUBLIC_REPO))
                 .replace('__PUBLIC_COMMIT__', repr(PUBLIC_COMMIT))
                 .replace('__PACK_DIR__', repr(PACK_DIR))
                 .replace('__THRESHOLDS__', repr(THRESHOLDS))
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
    # The stamp keeps a second run from colliding with the first one's job name.
    job_name = 'tancho-%s-%s' % (NAME, stamp.lower())
    key = RUN / 'ssh_key'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', NAME,
                        '-f', str(key)], check=True)
    key.chmod(0o600)
    command = [str(Path.home() / '.local/bin/nebius'), 'ai', 'job', 'create',
               '--parent-id', PROJECT, '--name', job_name,
               '--platform', 'gpu-h100-sxm', '--preset', '1gpu-16vcpu-200gb',
               '--image', IMAGE_REF,
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
        'job_name': job_name, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
        'framework_revision': FRAMEWORK_REVISION, 'image_ref': IMAGE_REF,
        'injected_bytes': injected, 'stamp': stamp, 'executed': args.execute}, indent=2) + '\n')
    print(completed.stdout.strip())
    print('injected bytes:', injected)


if __name__ == '__main__':
    main()
