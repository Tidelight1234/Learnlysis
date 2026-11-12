# Learnlysis

## PySide6 v5

This repository publishes **only the PySide6 v5 version** of Learnlysis. The PyQt5 version, PySide6 v6, text coder/trainer projects, local Python environments, build intermediates, and personal runtime data are excluded by `.gitignore`.

## Run the source version

Requires Python 3.11. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\v5\src\main.py
```

The optional video-frame backend can be installed with `requirements-ai.txt`.

## Run the Windows release

The ready-to-run onedir package is in `v5/dist/Learnlysis/`. Launch `Learnlysis.exe`; keep the neighboring `_internal` and license files with it. Per-user data is not part of the Git upload.

## Licenses

Learnlysis original code is licensed under the MIT License; see [LICENSE](LICENSE). Third-party libraries, Qt/PySide6, Python, fonts, and other bundled material remain under their own licenses; see [NOTICE](NOTICE), `LGPL-3.0.txt`, `LGPL-2.1.txt`, `PYTHON-LICENSE.txt`, and `THIRD_PARTY_LICENSES/`. The Windows release also carries these materials beside the executable. The Qt/PySide6 and FFmpeg source retrieval information and replacement rights are described in `NOTICE`.

