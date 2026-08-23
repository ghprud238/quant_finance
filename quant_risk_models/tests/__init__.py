import sys
from pathlib import Path
project_src = str(Path(__file__).resolve().parent.parent / 'src')
if project_src not in sys.path:
    sys.path.insert(0, project_src)
