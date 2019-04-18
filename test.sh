#!/usr/bin/env bash
pycodestyle --max-line-length=120 mxnet_octave_conv tests && \
    nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=mxnet_octave_conv tests
