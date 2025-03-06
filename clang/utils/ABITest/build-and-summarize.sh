<<<<<<< HEAD
#!/bin/sh

set -eu

if [ $# != 1 ]; then
    echo "usage: $0 <num-tests>"
    exit 1
fi

dir=$(dirname $0)
$dir/build.sh $1 &> /dev/null || true
../summarize.sh $1 &> fails-x.txt
cat fails-x.txt
wc -l fails-x.txt
=======
#!/bin/sh

set -eu

if [ $# != 1 ]; then
    echo "usage: $0 <num-tests>"
    exit 1
fi

dir=$(dirname $0)
$dir/build.sh $1 &> /dev/null || true
../summarize.sh $1 &> fails-x.txt
cat fails-x.txt
wc -l fails-x.txt
>>>>>>> 9a2a7a370a31 ([CIR][CUDA] Support for built-in CUDA surface type)
