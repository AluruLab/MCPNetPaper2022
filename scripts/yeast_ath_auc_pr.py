import sklearn
import argparse
import ast
import json
import sys
import sklearn.metrics as skm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def as_float(value):
    try:
        rx = float(value)
        return rx
    except ValueError:
        return 0.0


def compute_aupr_df1(gs_df, predict_df, plot=False, percentile=100):
    #print(gs_df.columns, gs_df.shape, predict_df.columns, predict_df.shape)
    ispercentile = (percentile is not None) and (percentile < 100) and (percentile > 0)
    if ispercentile:
        thresh = np.percentile(abs(predict_df.pwt), 100 - percentile)
    # 
    # Note that either Left/Outer join is fine here for this script because 
    # the predict_df is a expected to be a subnetwork of 
    # all the possible edges implied by gs_df.TF -- gs_df.TARGET
    #
    combo_df = gs_df.merge(predict_df, how='left', on = ["TF", "TARGET"])
    combo_df.fillna(0.0, inplace=True)
    if ispercentile:
        combo_df = combo_df[abs(combo_df.pwt) > thresh]
    roc_pr_stats = {
        'GS_SHAPE1': gs_df.shape[0], 
        'PREDICT_SHAPE1' :  predict_df[abs(predict_df.pwt) > thresh].shape[0] if (max(combo_df.pwt) != min(combo_df.pwt) and ispercentile) else predict_df.shape[0],
        'COMBO_SHAPE1' : combo_df.shape[0],
        #skm.roc_auc_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
        #skm.average_precision_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
        'AUROC1': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
        'AUPR1' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
       } 
    
    gs_df = gs_df.loc[gs_df.TF != gs_df.TARGET]
    predict_df = predict_df.loc[predict_df.TF != predict_df.TARGET]
    combo_df = gs_df.merge(predict_df, how='outer', on = ["TF", "TARGET"])
    combo_df.fillna(0.0, inplace=True)
    if ispercentile:
        combo_df = combo_df[abs(combo_df.pwt) > thresh]
    roc_pr_stats.update({
       'GS_UNIQ_SHAPE1' : gs_df.shape[0], 
       'PREDICT_UNIQ_SHAPE1' :  predict_df[abs(predict_df.pwt) > thresh].shape[0] if (max(combo_df.pwt) != min(combo_df.pwt) and ispercentile) else predict_df.shape[0],
       'COMBO_UNIQ_SHAPE1' : combo_df.shape[0],
       #skm.roc_auc_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       #skm.average_precision_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       'UNIQ_AUROC1': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
       'UNIQ_AUPR1' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
      })
    if plot is True:
        fpr, tpr, _ = skm.roc_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        precision, recall, _ = skm.precision_recall_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        return roc_pr_stats, fpr, tpr, precision, recall
    else:
        return roc_pr_stats, None, None, None, None

def compute_aupr_marbach(gs_df, predict_df):
    ## normalize weights between 0 and 1
    predict_df['pwt'] = abs(predict_df.pwt)
    pred_wt = predict_df['pwt']
    norm_pred_wt=(pred_wt-pred_wt.min())/(pred_wt.max()-pred_wt.min())
    predict_df['pwt'] = pred_wt
    predict_df = predict_df.nlargest(100000,columns=['pwt'],keep='all')
    #
    T = gs_df.shape[0]
    P = np.sum(gs_df.twt)
    N = T - P
    ##
    pdf = predict_df.merge(gs_df, on = ["TF", "TARGET"])
    L = pdf.shape[0]
    TPL = np.sum(pdf.twt) # true positives
    if L < T:
        p = float((P - TPL))/float((T-L))
    else:
        p = 0.0
    pdiscovery = pdf.twt.to_numpy()
    ndiscovery = 1 - pdf.twt.to_numpy()
    # find edges in gs missing in true edge
    pdf = pdf[['TF', 'TARGET', 'pwt']]
    combo_df = gs_df.merge(pdf, how='outer', on = ["TF", "TARGET"])
    MSNG = combo_df.pwt.isna().sum()
    #print(T, P, N, L, TPL, MSNG, p, pdf.shape, combo_df.shape)
    ##
    pos_random = np.repeat(p, MSNG)
    neg_random = np.repeat(1-p, MSNG)
    pos_discovery = np.concatenate((pdiscovery, pos_random))
    neg_discovery = np.concatenate((ndiscovery, neg_random))
    TPk = np.cumsum(pos_discovery)
    FPk = np.cumsum(neg_discovery)
    ##
    K = np.arange(1, T+1)
    TPR = TPk / P
    FPR = FPk / N
    REC = TPR
    PREC = TPk / K
    #
    AUROC = np.trapz(FPR, TPR)
    AUPR = np.trapz(PREC, REC) / (1-1/float(P))
    #print(AUROC, AUPR)
    return {
        'GS_SHAPE': gs_df.shape[0], 
        'PREDICT_SHAPE' :  N,
        'AUROC': AUROC,
        'AUPR' : AUPR
       }

def compute_aupr_df(gs_df, predict_df, plot, percentile=100):
    ##
    roc_pr_stats = {}
    rx = compute_aupr_marbach(gs_df, predict_df)
    roc_pr_stats.update(rx)
    ##
    tgs_df = gs_df.loc[gs_df.TF != gs_df.TARGET]
    rx2 = compute_aupr_marbach(tgs_df, predict_df)
    roc_pr_stats.update({'UNIQ_'+x : y for x,y in rx2.items()})
    vx, fpr, tpr, precision, recall =  compute_aupr_df1(gs_df, predict_df, plot, percentile=100)
    roc_pr_stats.update(vx)
    return roc_pr_stats, fpr, tpr, precision, recall


def compute_aupr(gs_file, predict_file):
    gs_df = pd.read_csv(gs_file, sep="\t", names=["TF", "TARGET", "twt"])
    predict_df = pd.read_csv(predict_file, sep="\t", names=["TF", "TARGET", "pwt"] )
    return compute_aupr_df(gs_df, predict_df)

def tsv_convert(network_file, gs_df):
    gene_wt = {}
    with open(network_file) as ifx:
        rlines = [lx.strip().split() for lx in ifx]
        gene_wt = {(x[0], x[1]): as_float(x[2]) for x in rlines}
    net_edges = []
    counts1 = 0
    counts2 = 0
    for row in gs_df.itertuples():
        #print(row, row.TF, row.TARGET)
        tf = row.TF
        gene = row.TARGET
        if (tf, gene) in gene_wt:
            counts1 += 1
            wt = gene_wt[(tf, gene)]
            net_edges.append((tf, gene, wt))
        elif (gene, tf) in gene_wt:
            counts1 += 1
            wt = gene_wt[(gene, tf)]
            net_edges.append((tf, gene, wt))
        else:
            counts2 += 1
    odf = pd.DataFrame(data=net_edges, columns=["TF", "TARGET", "pwt"])
    #print(odf.columns, odf.shape)
    #odf.to_csv(out_cmp_file, "\t", index=False, header=False)
    return odf


def matrix_covert(network_file, gs_df, tsep="\t", idx_col=None):
    #with open(gs_file) as gsx:
    #    glines = [x.strip().split() for x in gsx]

    df = pd.read_csv(network_file, sep=tsep, index_col=idx_col)
    tf_lst = list(y for y in dict.fromkeys([x for x in gs_df.TF]) if y in df.columns)
    gene_lst = list(y for y in dict.fromkeys([x for x in gs_df.TARGET]) if y in df.columns)

    print(df.shape, len(tf_lst), len(gene_lst))
    vals = df.loc[tf_lst,gene_lst].to_numpy().flatten()
    tfnames = [item for item in tf_lst for i in range(len(gene_lst))]
    genenames = gene_lst * len(tf_lst)

    odf = pd.DataFrame(data={'TF': tfnames, 'TARGET': genenames, 'pwt': vals})
    #odf.to_csv(out_cmp_file, "\t", index=False, header=False)
    return odf


def clr_convert(network_file, gs_file):
    return matrix_covert(network_file, gs_file, tsep="\t")

def mrnet_convert(network_file, gs_file):
    return matrix_covert(network_file, gs_file, tsep=" ")

def wgcna_convert(network_file, gs_file):
    return matrix_covert(network_file, gs_file, tsep=" ", idx_col = 0)

CONVERT_METHOD_DICT = {
    'aracne'  : tsv_convert,
    'arboreto' : tsv_convert,
    'clr'     : clr_convert,
    #'grnboost' : tsv_convert,
    'inferelator' : tsv_convert,
    'mrnet' : mrnet_convert,
    'wgcna' : wgcna_convert,
    'tinge' : tsv_convert,
}

def net_nrows(gs_df, predict_fxentry):
    method_name = predict_fxentry["method"]
    predict_file = predict_fxentry["file"]
    if method_name in CONVERT_METHOD_DICT:
        convert_fn = CONVERT_METHOD_DICT[method_name]
        odf = convert_fn(predict_file, gs_df)
        print(method_name, odf.shape)
        return odf.shape[0]
    else:
        print(method_name, 'is None')
        return None

def n_min_rows(gs_df, pfx_lst):
    nrow_lst = [net_nrows(gs_df, predict_fxentry) for predict_fxentry in pfx_lst]
    return min([x for x in nrow_lst if x is not None])

def plot_method_roc(gs_df, predict_fxentry, plot=False, nrows=None):
    method_name = predict_fxentry["method"]
    method_label = predict_fxentry["label"]
    predict_file = predict_fxentry["file"]
    pt = predict_fxentry["percentile"]
    prec = None
    recall = None
    if method_name in CONVERT_METHOD_DICT:
        convert_fn = CONVERT_METHOD_DICT[method_name]
        odf = convert_fn(predict_file, gs_df)
        if nrows is not None:
            odf = odf.nlargest(nrows, ['pwt'])
        apx, fpr, tpr, prec, recall = compute_aupr_df(gs_df, odf, plot, pt)
        apx.update({"label": method_label, "method": method_name})
        #print("\t".join(out_lst))
        print(method_name, odf.shape, apx)
        return apx, fpr, tpr, prec, recall 
    else:
        print(method_name + " is not found")
        return {}, None, None, None, None



def main(input_json):
    with open(input_json) as jfptr:
        jsx = json.load(jfptr)
    plt_colors = jsx['PLOT_COLORS']
    out_format = jsx['OUT_FORMAT']
    if out_format is not "none":
        plot = True
        matplotlib.style.use('seaborn-muted')
        fig_size=ast.literal_eval(jsx['FIG_SIZE'])
        subfig_size=ast.literal_eval(jsx['SUB_FIG_SIZE'])
    else:
        plot = False
    if 'PERCENTILE' in jsx:
        pt = int(jsx['PERCENTILE'])
    else: 
        pt = 100 
    ncolors = len(plt_colors)
    gs_file = jsx['DATA_DIR'] + "/" + jsx['TRUE_NET']
    gs_df = pd.read_csv(gs_file, sep="\t", names=["TF", "TARGET", "twt"])
    out_file = jsx["OUT_FILE"]
    rocpr_out_file = jsx['ROCPR_OUTPUT_FILE']+out_format
    print("Out format      :  ", out_format)
    print("Fig Size        : ", jsx['FIG_SIZE'])
    print("Plot            : ", plot)
    print("Percentile      : ", pt)
    print("GS File         : ", gs_file)
    print("Out File        : ", out_file)
    print("ROCPR Out File  : ", rocpr_out_file)
    pfx_lst = []
    for idx, file_entry in enumerate(jsx['REVNET_FILES']):
        pfx_entry = {}
        pfx_entry["method"] = file_entry["method"]
        pfx_entry["file"] = jsx['DATA_DIR'] +  "/" + file_entry["file"]
        pfx_lst.append(pfx_entry)
    nrows = n_min_rows(gs_df, pfx_lst)
    roc_pr_stats = {}
    out_df_lst = []
    for idx, file_entry in enumerate(jsx['REVNET_FILES']):
        pfx_entry = {}
        pfx_entry["method"] = file_entry["method"]
        pfx_entry["label"] = file_entry["label"]
        pfx_entry["file"] = jsx['DATA_DIR'] +  "/" + file_entry["file"]
        pfx_entry["percentile"] = pt
        rdict, fpr, tpr, prec, recall = plot_method_roc(gs_df, pfx_entry, plot, None)
        if plot is True:
            roc_pr_stats[file_entry["label"]] = [fpr, tpr, prec, recall]
        out_df_lst.append(rdict)
    odf = pd.DataFrame(out_df_lst)
    odf.to_csv(out_file, index=False)
    # diagonal line
    if plot is True:
        fig = plt.figure(figsize=subfig_size)
        plt.subplot(1, 2, 1)
        for idx, krpr in enumerate(roc_pr_stats.items()):
            plot_color = plt_colors[idx % ncolors]
            method_label, rpr = krpr
            fpr, tpr, _, _ = rpr
            if fpr is None:
                print(method_label + "is None", krpr)
            plt.plot(fpr, tpr, color=plot_color, label=method_label)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        #
        #
        plt.subplot(1, 2, 2)
        for idx, krpr in enumerate(roc_pr_stats.items()):
            plot_color = plt_colors[idx % ncolors]
            method_label, rpr = krpr
            _, _, prec, recall = rpr
            if prec is None:
                print(method_label + "is None")
            plt.plot(prec, recall, color=plot_color, label=method_label)
        plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='upper right')
        fig.savefig(rocpr_out_file)


if __name__ == "__main__":
    #    f1 = sys.argv[1]
    #    f2 = sys.argv[2]
    PROG_DESC = """ ROC/PR of Real Networks"""
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("-f", "--input_json", default="ecoli_options.json",
                       help="Simulated Input Files")
    ARGS = PARSER.parse_args()
    print("""ARG : JSON File : %s """ % (ARGS.input_json))
    main(ARGS.input_json)
 
