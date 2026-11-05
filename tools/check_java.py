"""Parse Java syntax without installing Android SDK or compiling an APK."""
from pathlib import Path
import javalang

root=Path(__file__).resolve().parents[1]
files=list((root/'android'/'app'/'src'/'main'/'java').rglob('*.java'))
for path in files:
    try:javalang.parse.parse(path.read_text(encoding='utf-8'))
    except javalang.parser.JavaSyntaxError as error:
        raise SystemExit(f'{path}: {error.description}; token {error.at}') from error
    print('Java syntax OK:',path.name)
print(f'{len(files)} Java files parsed. This does not check Android API types or create an APK.')
