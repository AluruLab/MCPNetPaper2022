#!/bin/bash

# get all the combo aupr lines from eval_micombnet

# get all the "Computed" lines, and use Output line to extract iteration and dataset name
# then narrow it down to MI, DPI, aupr, Combo (need all combo since stouffer is in there too)
# then exclude the mi, dpi1, dpi2 and dpi3 calcs in TF calls as they are usefull only for mem and time (separately extracted)
grep -P "^Computed,MAX Combo TF|^Computing," *_tf_*_eval5.log | grep -v "Stouffer" > train_test_aupr_violin_tf.tmp.txt
# exclude stouffer

# change percent from double to someting more readable.
cat train_test_aupr_violin_tf.tmp.txt | \
   sed -e 's#percent 0\.499[0-9]*,#percent 0.5,#g' | \
   sed -e 's#percent 0\.999[0-9]*,#percent 1,#g' | \
   sed -e 's#percent 1\.999[0-9]*,#percent 2,#g' | \
   sed -e 's#percent 4\.999[0-9]*,#percent 5,#g' | \
   sed -e 's#percent 9\.999[0-9]*,#percent 10,#g' | \
   sed -e 's#percent 19\.999[0-9]*,#percent 20,#g' | \
   sed -e 's#percent 29\.999[0-9]*,#percent 30,#g' | \
   sed -e 's#percent 39\.999[0-9]*,#percent 40,#g' | \
   sed -e 's#percent 49\.999[0-9]*,#percent 50,#g' \
   > train_test_aupr_violin_tf.txt

rm train_test_aupr_violin_tf.tmp.txt

# # for MI, DPI1, DPI2, DPI3:   merge compute and aupr lines.
echo "percent,iterations,method,partition,aupr,combo,aupr_time" > train_test_aupr_violin_tf.csv



# for combo and stouffer, we will have Combo, aupr, Combo AUPR lines.  ignore aupr line.

# get dataset and run id.
# get the method type
# convert maxmin tag to combo
# parse MI
# parse DPI 1, 2, 3
# parse combo FULL and TF.
# parse combo stouffer summary.
# fix a typo with stouffer test stdev.
# convert decile to percent
# the fill foward
# and remove the methods lines.
# rename DPI to MCP
# merge orig, and merge stouffer.
# default first run as 1.
cat train_test_aupr_violin_tf.txt | \
   sed -e 's#^\([^_]*\)_[^\.]*_eval\([0-9]*\)\.log:#\1,\2,#g' | \
   sed -e 's#Computing,percent \([^,]*\),iters \([^ ]*\),sec$#\1,\2,#g' | \
   sed -e 's#Computing, TF percent \([^,]*\),iters \([^ ]*\),sec$#\1,\2,#g' | \
   sed -e 's#Computed,MAX Combo TF aupr_\([^ ]*\) \([0-9\.]*\),\(.*\),sec$#,,Combo,\1,\2,\3#g' | \
   sed -e 's#Computed,MAX Combo AND Stouffer TF aupr_\([^ ]*\) \([0-9\.]*\),\(.*\),sec$#,,Combo Stouffer,\1,\2,\3#g' | \
   awk -F',' -v COL=1 '$COL=="" {$COL = saved} { OFS=","; saved = $COL; print}' | \
   awk -F',' -v COL=2 '$COL=="" {$COL = saved} { OFS=","; saved = $COL; print}' | \
   sed -e "/^[0-9\.]*,[0-9]*,$/d" \
>> train_test_aupr_violin_tf.csv

# NOTE: will need to adjust the mean and stdev AUPR to NOT use psuedocount.
# do in excel.

