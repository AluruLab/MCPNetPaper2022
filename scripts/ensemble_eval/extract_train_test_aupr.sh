#!/bin/bash

# get all the "Computed" lines, and use Output line to extract iteration and dataset name
# then narrow it down to MI, DPI, aupr, Combo (need all combo since stouffer is in there too)
# then exclude the mi, dpi1, dpi2 and dpi3 calcs in TF calls as they are usefull only for mem and time (separately extracted)
grep -P "^AUPR|,MI,|,DPI|,MAXMIN" *_eval5.log > train_test_aupr2.txt

echo "organism,tf,run,method,partition,stat,aupr,part_size,iterations" > train_test_aupr2.csv

# # for MI, DPI1, DPI2, DPI3:   merge compute and aupr lines.



# for combo and stouffer, we will have Combo, aupr, Combo AUPR lines.  ignore aupr line.

# get dataset and run id.
# get the method type
# convert maxmin tag to combo
# parse MI
# parse DPI 1, 2, 3
# parse combo FULL and TF.
# parse combo stouffer summary.
# fix a typo with stouffer test stdev.  not stdev may be in scientific notation
# convert decile to percent
# the fill foward
# and remove the methods lines.
# rename DPI to MCP
# merge orig, and merge stouffer.
# default first run as 1.
cat train_test_aupr2.txt | \
   sed -e 's#^\([^_]*\)_\([^_]*\)_[^\.]*_eval\([0-9]*\)\.log:#\1,\2,\3,#g' | \
   sed -e 's#Computed,\([^,]*\),.*$#\1,#g' | \
   sed -e 's#MAXMIN 1 2 3,#Combo,#g' | \
   sed -e 's#,AUPR TF #,AUPR #g' | \
   sed -e 's#,AUPR stouffer TF #,AUPR stouffer #g' | \
   sed -e 's#,AUPR,\([0-9\.]*\),,,sec$#,,orig,FULL,val,\1,100,1#g' | \
   sed -e 's#,AUPR TF,\([0-9\.]*\),,,sec$#,,orig,TF,val,\1,TF,1#g' | \
   sed -e 's#,AUPR Not_TF,\([0-9\.]*\),,,sec$#,,orig,~TF,val,\1,~TF,1#g' | \
   sed -e 's#,AUPR combo FULL,\([0-9\.]*\),,,sec$#,,orig,FULL,val,\1,100,1#g' | \
   sed -e 's#,AUPR combo TF,\([0-9\.]*\),,,sec$#,,orig,TF,val,\1,TF,1#g' | \
   sed -e 's#,AUPR combo Not_TF,\([0-9\.]*\),,,sec$#,,orig,~TF,val,\1,~TF,1#g' | \
   sed -e 's#,AUPR stouffer combo FULL,\([0-9\.]*\),,,sec$#,,st,FULL,val,\1,100,1#g' | \
   sed -e 's#,AUPR stouffer combo TF,\([0-9\.]*\),,,sec$#,,st,TF,val,\1,TF,1#g' | \
   sed -e 's#,AUPR stouffer combo Not_TF,\([0-9\.]*\),,,sec$#,,st,~TF,val,\1,~TF,1#g' | \
   sed -e 's#,decile 0,#,percent 0.5,#g' | \
   sed -e 's#,decile 1,#,percent 1,#g' | \
   sed -e 's#,decile 2,#,percent 2,#g' | \
   sed -e 's#,decile 3,#,percent 5,#g' | \
   sed -e 's#,decile 4,#,percent 10,#g' | \
   sed -e 's#,decile 5,#,percent 20,#g' | \
   sed -e 's#,decile 6,#,percent 30,#g' | \
   sed -e 's#,decile 7,#,percent 40,#g' | \
   sed -e 's#,decile 8,#,percent 50,#g' | \
   sed -e 's#,AUPR \([^ ]*\) \([^ ]*\),\([0-9\.]*[0-9e\-]*\),percent \([0-9\.]*\),iters \([0-9\.]*\),sec$#,,orig,\1,\2,\3,\4,\5#g' | \
   sed -e 's#,AUPR stouffer \([^ ]*\) \([^ ]*\),\([0-9\.]*[0-9e\-]*\),percent \([0-9\.]*\),iters \([0-9\.]*\),sec$#,,st,\1,\2,\3,\4,\5#g' | \
   perl -0pe 's/,test,mean,([^\n]*)\n([^\n]*),train,stdev,/,test,mean,$1\n$2,test,stdev,/g' | \
   awk -F',' -v COL=4 '$COL=="" {$COL = saved} { OFS=","; saved = $COL; print}' | \
   sed -e "/^[^,]*,[^,]*,[0-9]*,[^,]*,$/d" | \
   sed -e "s/DPI/MCP/g" | \
   sed -e 's/,Combo,st,/,Stouffer,/g' | \
   sed -e 's/,orig,/,/g' | \
   sed -e 's/^gnw2000,,/gnw2000,1,/g' \
   >> train_test_aupr2.csv


# NOTE: will need to adjust the mean and stdev AUPR to NOT use psuedocount.
# do in excel.