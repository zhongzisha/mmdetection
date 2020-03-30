#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CURDIR=`pwd`

echo "Building roi align rotated op..."
cd mmdet/ops/roi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
cd $CURDIR

echo "Building ps roi align rotated op..."
cd mmdet/ops/psroi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
cd $CURDIR

echo "Building poly_nms op..."
cd mmdet/ops/poly_nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
cd $CURDIR

echo "Building cpu_nms..."
cd mmdet/core/bbox
$PYTHON setup_linux.py build_ext --inplace
cd $CURDIR


