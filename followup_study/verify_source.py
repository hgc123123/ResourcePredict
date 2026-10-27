"""Verify this supplement's source files without dependencies or shell quoting."""
from __future__ import annotations
import hashlib
import argparse
from pathlib import Path
import sys


def main(argv=None):
    root=Path(__file__).resolve().parent
    manifest=root/'SOURCE_SHA256SUMS.txt'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',
                        help='Generate this distribution source manifest; not an attestation of historical execution')
    args=parser.parse_args(argv)
    if args.write:
        paths=sorted(p for p in root.rglob('*') if p.is_file()
                     and p.name!='SOURCE_SHA256SUMS.txt'
                     and p.suffix in ('.py','.md','.txt','.slurm','.json')
                     and not any(part in ('__pycache__','qa_tmp','outputs') for part in p.relative_to(root).parts))
        lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix()
               for p in paths]
        manifest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f'Wrote distribution manifest for {len(lines)} source files.')
        return 0
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    failed=[]
    count=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest,name=line.split('  ',1)
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failed.append(name)
            continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        count+=1
        if actual!=digest:
            failed.append(name)
    if failed:
        print('Source verification failed: '+', '.join(failed),file=sys.stderr)
        return 1
    print(f'OK: {count} source files verified. This does not verify experiment results.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
