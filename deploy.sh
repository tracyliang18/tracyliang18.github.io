#!/bin/bash
# Created Time: Mon 16 Oct 2017 09:15:45 PM HKT
# Author: Jiajun Liang
# Mail: liangjiajun@megvii.com

set -x

BASEDIR=$(dirname "$0")


# hack(copy imgs for relative issue img src="imgs...") and view locally
for img_dir in $(find ${BASEDIR} -type d -mindepth 2 -name imgs)
do
    cp ${img_dir}/* ${BASEDIR}/imgs/
done

bundle exec jekyll build -d ~/www
