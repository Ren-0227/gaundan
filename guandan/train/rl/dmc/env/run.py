import os
import subprocess

procs = []
for i in range(1, 17):                 # 1..8
    cwd = f'GdAITest_package{i}/linux'
    if not os.path.isfile(os.path.join(cwd, 'GdAITest')):
        print(f'跳过：{cwd}/GdAITest 不存在')
        continue
    # 启动子进程，不阻塞
    p = subprocess.Popen(['./GdAITest'], cwd=cwd)
    procs.append(p)

# 可选：等待全部结束
for p in procs:
    p.wait()
