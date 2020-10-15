CONFIG_PREFIX=$1
bash train1.sh ${CONFIG_PREFIX}
bash parse_results.sh ${CONFIG_PREFIX}
cd DOTA_devkit
bash do1.sh ${CONFIG_PREFIX}
bash demo_large_OBB.sh ${CONFIG_PREFIX}
