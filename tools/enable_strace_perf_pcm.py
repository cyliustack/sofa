#!/usr/bin/env python
import subprocess
subprocess.call('sudo sysctl -w kernel.yama.ptrace_scope=0', shell=True)
subprocess.call('sudo sysctl -w kernel.perf_event_paranoid=-1', shell=True)
subprocess.call('sudo sysctl -w kernel.kptr_restrict=0', shell=True)
