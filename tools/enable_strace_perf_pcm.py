#!/usr/bin/env python
import subprocess
subprocess.call('sudo sysctl -w kernel.yama.ptrace_scope=0', shell=True)
subprocess.call('sudo sysctl -w kernel.nmi_watchdog=0', shell=True)
subprocess.call('sudo modprobe msr', shell=True)
subprocess.call('sudo sysctl -w kernel.perf_event_paranoid=-1', shell=True)
subprocess.call('sudo sysctl -w kernel.kptr_restrict=0', shell=True)
subprocess.call('sudo setcap cap_sys_rawio=ep `which pcm-pcie.x`', shell=True)
subprocess.call('sudo setcap cap_sys_rawio=ep `which pcm-numa.x`', shell=True)
subprocess.call('sudo setcap cap_sys_rawio=ep `which pcm-memory.x`', shell=True)
subprocess.call('sudo setcap cap_sys_rawio=ep `which pcm-numa.x`', shell=True)
