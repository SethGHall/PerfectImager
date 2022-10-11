#!/bin/bash

COMMAND=${1}

PROJECT_DIR=${PWD}
BUILD_DIR=${PROJECT_DIR}/build
CONFIG_DIR=${PROJECT_DIR}/configs

GLEAM_SMALL=${CONFIG_DIR}/gleam_small.yaml
GAUSSIAN=${CONFIG_DIR}/gaussian.yaml

case ${COMMAND} in

    build_and_run_gleam_small)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./direct_imaging ${GLEAM_SMALL}
        cd ${PROJECT_DIR}
    ;;
	
	build_and_run_gaussian)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./direct_imaging ${GAUSSIAN}
        cd ${PROJECT_DIR}
    ;;

    *)
        echo "ERR: Unrecognized command, please review the script for valid commands..."
    ;;

esac

