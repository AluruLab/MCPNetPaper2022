#!/bin/bash

# get all the "Computed" lines, and use Output line to extract iteration and dataset name
# then narrow it down to MI, DPI, aupr, Combo (need all combo since stouffer is in there too)
# then exclude the mi, dpi1, dpi2 and dpi3 calcs in TF calls as they are usefull only for mem and time (separately extracted)
grep -P "Computed," *_mcpnet*.log | grep -P "MI,|DPI|aupr |Combo|MAXMIN 1 2 3" > gnw2000_aupr.txt

echo "organism,TF,run,algo,combo,algo_time,aupr,aupr_time" > gnw2000_aupr.csv

# for MI, DPI1, DPI2, DPI3:   merge compute and aupr lines.
# for combo and stouffer, we will have Combo, aupr, Combo AUPR lines.  ignore aupr line.

# remove the rank tag
# then extract the iteration (memX) and dataset name.  subsetquent entries will ned to fill from thee.
# the parse the filenames,
# replace "Computed" and leave 2 spaces to align to iteration and dataset.
# remove trailing sec
# merge MI, DPI, and their following aupr lines.
# then merge Combo and Combo AUPR, skiping the "aupr" line in between.
# then merge Combo Stouffer and Combo Stouffer AUPR, skiping the "aupr" line in between.
# then merge Max Combo and MAX Combo AND Stouffer, also skipping the "aupr" line in between
# finally remove the second combo field
# then remove the commandline. lines.
cat gnw2000_aupr.txt | \
   sed -e 's#\[[0-9]*\] ##g' | \
   sed -e 's#gnw2000_\([^_]*\)_mcpnet\([0-9]*\)\.log:#gnw2000,\1,\2,#g' | \
   sed -e 's#Computed,##g' | \
   sed -e 's#,sec##g' | \
   sed -e 's#,MAXMIN 1 2 3,,\([0-9\.]*\)$#,MAXMIN,,\1,,,#g' | \
   perl -0pe 's/,MI,([^\n]*)\n[^\n]*aupr /,MI,$1,/g' | \
   perl -0pe 's/,DPI1,([^\n]*)\n[^\n]*aupr /,DPI1,$1,/g' | \
   perl -0pe 's/,DPI2,([^\n]*)\n[^\n]*aupr /,DPI2,$1,/g' | \
   perl -0pe 's/,DPI3,([^\n]*)\n[^\n]*aupr /,DPI3,$1,/g' | \
   perl -0pe 's/,Combo,([^\n]*)\n[^\n]*aupr [^\n]*\n[^\n]*Combo AUPR /,Combo,$1,/g' | \
   perl -0pe 's/,Combo Stouffer,([^\n]*)\n[^\n]*aupr [^\n]*\n[^\n]*Combo Stouffer AUPR /,Combo Stouffer,$1,/g' | \
   perl -0pe 's/,MAX Combo aupr [0-9\.]*,([^\n]*)\n[^\n]*aupr [^\n]*\n[^\n]*MAX Combo aupr auroc /,MAX Combo,$1,/g' | \
   perl -0pe 's/,MAX Combo AND Stouffer aupr [0-9\.]*,([^\n]*)\n[^\n]*aupr [^\n]*\n[^\n]*MAX Combo AND Stouffer aupr auroc /,MAX Combo Stouffer,$1,/g' | \
   sed -e "s/\([0-9\.]*\),[()0-9\._]*,\([0-9\.]*\)$/\1,\2/g" \
   >> gnw2000_aupr.csv

# need to fill downward the iteration and dataset -  do in excel.
#
#
