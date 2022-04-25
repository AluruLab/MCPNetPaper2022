#
# Copyright 2021 Georgia Tech Research Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Author(s): Tony C. Pan
#

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(version = "3.10")  # 3.10 needed for R 3.6.x 
# need apt install libcurl4-oepnssl-dev libssl-dev libgit2.dev libxml2-dev  
install.packages("devtools")
library(devtools)
install_url("https://cran.r-project.org/src/contrib/Archive/PCIT/PCIT_1.5-3.tar.gz")

# need to run .libPaths() for the path below.
#BiocManager::install("netbenchmark", lib="/home/tpan/R/x86_64-pc-linux-gnu-library/3.6")
#library(netbenchmark)

# http://bioconductor.org/packages/release/data/experiment/manuals/grndata/man/grndata.pdf
BiocManager::install("grndata")
library(grndata)

install.packages("data.table")
library(data.table)

ndata = length(Availabledata)

# get the data out.  (see netbenchmark source code)
for ( n in seq_len(ndata)) {
    ### get data
    data = grndata::getData(Availabledata[n], getNet=TRUE)
    print(length(data[[1]]))

    ### transpose it.
    # datat = transpose(as.data.frame(data[[1]]))
    datat = as.data.frame(t(as.matrix(data[[1]])))
    rownames(datat) = colnames(data[[1]])
    colnames(datat) = rownames(data[[1]])

    # add a gene alias column
    new_cols = append(colnames(datat), c('Alias'), 0)
    datat$Alias = "---"
    datat = datat[, new_cols]
    
    # write out
    write.table(datat, paste(Availabledata[n], "exp", sep="."), row.names=TRUE, col.names=TRUE, quote=FALSE, sep="\t", dec=".")
    write.table(data[[2]], paste(Availabledata[n], "net", sep="."), row.names=TRUE, col.names=TRUE, quote=FALSE, sep=" ", dec=".")
}
