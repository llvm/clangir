<<<<<<< HEAD
#!/usr/bin/env python3

import os, sys, subprocess


idx = 1
for arg in sys.argv[1:]:
  if not "=" in arg:
    break
  idx += 1
  (argname, argval) = arg.split("=")
  os.environ["SIMCTL_CHILD_" + argname] = argval

exitcode = subprocess.call(sys.argv[idx:])
if exitcode > 125:
  exitcode = 126
sys.exit(exitcode)
=======
#!/usr/bin/env python3

import os, sys, subprocess


idx = 1
for arg in sys.argv[1:]:
  if not "=" in arg:
    break
  idx += 1
  (argname, argval) = arg.split("=")
  os.environ["SIMCTL_CHILD_" + argname] = argval

exitcode = subprocess.call(sys.argv[idx:])
if exitcode > 125:
  exitcode = 126
sys.exit(exitcode)
>>>>>>> 9a2a7a370a31 ([CIR][CUDA] Support for built-in CUDA surface type)
