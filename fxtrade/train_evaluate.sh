#!/bin/sh

testdatafile="./fx-outdata-marge_uptrend.csv"
outputfile="result_uptrend.out"

nohup python fx_train.py ${testdatafile} > ${outputfile} &

pid1=$!

wait $pid1

nohup python fx_evaluate.py 20160128 20160129 > ${outputfile} &

pid1=$!

wait $pid1

nohup python fx_evaluate.py 20160129 20160201 > ${outputfile} &

pid1=$!

wait $pid1

nohup python fx_evaluate.py 20160202 20160203 > ${outputfile} &

pid4=$!

wait $pid4

nohup python fx_evaluate.py 20160203 20160204 > ${outputfile} &

pid5=$!

wait $pid5

nohup python fx_evaluate.py 20161107 20161108 > ${outputfile} &

pid6=$!

wait $pid6

nohup python fx_evaluate.py 20161108 20161109 > ${outputfile} &

pid7=$!

wait $pid7

nohup python fx_evaluate.py 20161109 20161110 > ${outputfile} &

pid8=$!

wait $pid8

nohup python fx_evaluate.py 20161110 20161111 > ${outputfile} &

pid9=$!

wait $pid9

nohup python fx_evaluate.py 20161111 20161114 > ${outputfile} &


