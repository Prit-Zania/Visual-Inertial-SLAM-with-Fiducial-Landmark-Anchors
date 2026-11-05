"""Package project source and demo; exclude environments, caches and this archive."""
from pathlib import Path
import hashlib
import zipfile

root=Path(__file__).resolve().parents[1]
destination=root/'OnePlus12R_MSCKF_Source.zip'
exclude={'.venv','__pycache__','.git','.gradle','build','recordings'}
files=[p for p in root.rglob('*') if p.is_file() and p!=destination and not any(part in exclude for part in p.relative_to(root).parts)
       and p.suffix not in ('.pyc','.zip')]
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for p in files:archive.write(p,p.relative_to(root))
with zipfile.ZipFile(destination) as archive:
    if archive.testzip() is not None:raise RuntimeError('Archive CRC check failed')
print(destination)
print('Files:',len(files),'SHA256:',hashlib.sha256(destination.read_bytes()).hexdigest())
