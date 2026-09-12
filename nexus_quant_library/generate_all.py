import os
import numpy as np
import pandas as pd

def write_module(rel_path, content):
    full = os.path.join('/working_dir/nexus_quant_platform', rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f_out:
        f_out.write(content.strip() + '\n')
    print('Generated:', rel_path)

print('Starting script generation...')
