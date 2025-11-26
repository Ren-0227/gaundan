import shutil
import json
from pathlib import Path

src_dir   = Path('../GdAITest_package')
base_port = 23455

for i in range(1, 17):                     
    dst_dir = Path(f'GdAITest_package{i}')
    if dst_dir.exists():                   # 已存在就删
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # 改 config.json
    cfg_file = dst_dir / 'linux' / 'AIConfig.json'
    if cfg_file.is_file():
        data = json.loads(cfg_file.read_text(encoding='utf-8'))

        # ===== 只改 wsEndpoint 端口 =====
        data['gameSettings']['wsEndpoint'] = f'http://localhost:{base_port + i}/'

        cfg_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding='utf-8')
    else:
        print(f'警告：{cfg_file} 不存在，跳过修改')
