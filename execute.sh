#!/bin/bash

COMMAND=${1}

PROJECT_DIR=${PWD}
BUILD_DIR=${PROJECT_DIR}/build
CONFIG_DIR=${PROJECT_DIR}/configs

GLEAM_SMALL=${CONFIG_DIR}/gleam_small.yaml
GLEAM_LARGE=${CONFIG_DIR}/gleam_large.yaml
GAUSSIAN=${CONFIG_DIR}/gaussian.yaml
VLA=${CONFIG_DIR}/vla.yaml

case ${COMMAND} in

    build_and_run_gleam_small)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./perfect_imager ${GLEAM_SMALL}
        cd ${PROJECT_DIR}
    ;;
	
    build_and_run_gleam_large)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./perfect_imager ${GLEAM_LARGE}
        cd ${PROJECT_DIR}
    ;;

    build_and_run_vla)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./perfect_imager ${VLA}
        cd ${PROJECT_DIR}
    ;;


	build_and_run_gaussian)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        ./perfect_imager ${GAUSSIAN}
        cd ${PROJECT_DIR}
    ;;

    *)
        echo "ERR: Unrecognized command, please review the script for valid commands..."
    ;;

esac
