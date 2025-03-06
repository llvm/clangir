<<<<<<< HEAD
#!/bin/sh

set -eu

if [ $# != 1 ]; then
    echo "usage: $0 <num-tests>"
    exit 1
fi

CPUS=2
make -j $CPUS \
  $(for i in $(seq 0 $1); do echo test.$i.report; done) -k
=======
#!/bin/sh

set -eu

if [ $# != 1 ]; then
    echo "usage: $0 <num-tests>"
    exit 1
fi

CPUS=2
make -j $CPUS \
  $(for i in $(seq 0 $1); do echo test.$i.report; done) -k
>>>>>>> 9a2a7a370a31 ([CIR][CUDA] Support for built-in CUDA surface type)
