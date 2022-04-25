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
import functools as ft
import numpy as np
import scipy.stats as stats
import math
import numpy.ma as ma

# with dagonal as is.
def stouffer(arr):
    zc_col = stats.zscore(arr, axis=0, ddof=1)
    zc_row = stats.zscore(arr, axis=1, ddof=1)

    st = (zc_row + zc_col) / np.sqrt(2)
    np.fill_diagonal(st, 0.0)

    return st

# with diagonal excluded
def stouffer2(values):
    diag = np.zeros(values.shape, dtype=bool)
    np.fill_diagonal(diag, True)
    arr = ma.array(data=values, mask=diag)
    zc_col = stats.zscore(arr, axis=0, ddof=1)
    zc_row = stats.zscore(arr, axis=1, ddof=1)

    st = ((zc_row + zc_col) / np.sqrt(2)).data.copy()
    np.fill_diagonal(st, 0.0)

    return st

# with diagonal set to 0
def stouffer3(values):
    arr = values.copy()
    np.fill_diagonal(arr, 0)
    zc_col = stats.zscore(arr, axis=0, ddof=1)
    zc_row = stats.zscore(arr, axis=1, ddof=1)

    st = (zc_row + zc_col) / np.sqrt(2)
    np.fill_diagonal(st, 0.0)

    return st


def make_stouffer(cdf):
    cm = cdf.to_numpy().copy()
    vx = stouffer2(cm)
    return pd.DataFrame(vx, columns=cdf.columns, index=cdf.index)

def as_float(value):
    try:
        rx = float(value)
        return rx
    except ValueError:
        return 0.0

def compute_aupr_df1(gs_df, predict_df, plot, percentile=100):
    #print(gs_df.shape, predict_df.shape)
    ispercentile = (percentile is not None) and (percentile < 100) and (percentile > 0)
    if ispercentile:
        thresh = np.percentile(abs(predict_df.pwt), 100 - percentile)
    predict_df = predict_df.sort_values(by="pwt", ascending=False)
    predict_df = predict_df.drop_duplicates(subset=["TF", "TARGET"])
    combo_df = gs_df.merge(predict_df, on = ["TF", "TARGET"], how="left")
    combo_df.fillna(0.0, inplace=True)
    #print("COMB", combo_df.isna().sum().sum(), np.isinf(combo_df.pwt.to_numpy()).sum(), np.isinf(combo_df.twt.to_numpy()).sum())
    #print(max(combo_df.pwt), min(combo_df.pwt), combo_df.shape)
    if max(combo_df.pwt) != min(combo_df.pwt) and ispercentile:
        combo_df = combo_df[abs(combo_df.pwt) > thresh]
    #print(max(combo_df.pwt), min(combo_df.pwt), combo_df.shape)
    roc_pr_stats = {
        'GS_SHAPE1': gs_df.shape[0], 
        'PREDICT_SHAPE1' :  predict_df[abs(predict_df.pwt) > thresh].shape[0] if (max(combo_df.pwt) != min(combo_df.pwt) and ispercentile) else predict_df.shape[0],
        'COMBO_SHAPE1' : combo_df.shape[0],
        #'AUROC1': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
        #'AUPR1' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        'AUROC1': skm.roc_auc_score(combo_df.twt, combo_df.pwt.to_numpy()),
        'AUPR1' : skm.average_precision_score(combo_df.twt, combo_df.pwt.to_numpy())
       } 
    gs_df = gs_df.loc[gs_df.TF != gs_df.TARGET]
    predict_df = predict_df.loc[predict_df.TF != predict_df.TARGET]
    combo_df = gs_df.merge(predict_df, on = ["TF", "TARGET"], how="left")
    #print("COMB", combo_df.head())
    combo_df.fillna(0.0, inplace=True)
    if max(combo_df.pwt) != min(combo_df.pwt) and ispercentile:
        combo_df = combo_df[abs(combo_df.pwt) > thresh]
    roc_pr_stats.update({
       'GS_UNIQ_SHAPE1' : gs_df.shape[0], 
       'PREDICT_UNIQ_SHAPE1' :  predict_df[abs(predict_df.pwt) > thresh].shape[0] if (max(combo_df.pwt) != min(combo_df.pwt) and ispercentile) else predict_df.shape[0],
       'COMBO_UNIQ_SHAPE1' : combo_df.shape[0],
       #skm.roc_auc_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       #skm.average_precision_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       #'UNIQ_AUROC1': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
       #'UNIQ_AUPR1' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
       'UNIQ_AUROC1': skm.roc_auc_score(combo_df.twt, combo_df.pwt.to_numpy()),
       'UNIQ_AUPR1' : skm.average_precision_score(combo_df.twt, combo_df.pwt.to_numpy())
      })
    if plot is True:
        fpr, tpr, _ = skm.roc_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        precision, recall, _ = skm.precision_recall_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        return roc_pr_stats, fpr, tpr, precision, recall
    else:
        return roc_pr_stats, None, None, None, None



def compute_aupr_df2(gs_df, predict_df, plot, percentile=100):
    tmp_df = gs_df
    predict_df['pwt'] = abs(predict_df.pwt)
    pred_wt = predict_df['pwt']
    norm_pred_wt=(pred_wt-pred_wt.min())/(pred_wt.max()-pred_wt.min())
    predict_df['pwt'] = pred_wt
    nselect = 100000
    predict_df = predict_df.nlargest(100000,columns=['pwt'],keep='all')
    ##
    pdf = gs_df.merge(predict_df, on = ["TF", "TARGET"])
    TP = np.sum(pdf.twt)
    FP = nselect - np.sum(pdf.twt)
    #print(predict_df.shape, pdf.shape, TP, FP)
    tratio=TP/(TP+FP)
    pdf = pdf[['TF', 'TARGET', 'pwt']]
    #print(pdf.head())
    combo_df = gs_df.merge(pdf, how='outer', on = ["TF", "TARGET"])
    combo_df.fillna(tratio, inplace=True)
    roc_pr_stats = {
        'GS_SHAPE': gs_df.shape[0], 
        'PREDICT_SHAPE' :  (TP+FP),
        #skm.roc_auc_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
        #skm.average_precision_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
        'AUROC': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
        'AUPR' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
       } 
    ##
    gs_df = tmp_df
    gs_df = gs_df.loc[gs_df.TF != gs_df.TARGET]
    pdf = gs_df.merge(predict_df, on = ["TF", "TARGET"])
    TP = np.sum(pdf.twt)
    FP = nselect - np.sum(pdf.twt)
    #print(predict_df.shape, pdf.shape, TP, FP)
    tratio=TP/(TP+FP)
    pdf = pdf[['TF', 'TARGET', 'pwt']]
    #print(pdf.head())
    combo_df = gs_df.merge(pdf, how='outer', on = ["TF", "TARGET"])
    combo_df.fillna(tratio, inplace=True)
    roc_pr_stats.update({
       'GS_UNIQ_SHAPE' : gs_df.shape[0], 
       'PREDICT_UNIQ_SHAPE' :  (TP+FP),
       #skm.roc_auc_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       #skm.average_precision_score(gs_df.twt, np.abs(predict_df.pwt.to_numpy())),
       'UNIQ_AUROC': skm.roc_auc_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy())),
       'UNIQ_AUPR' : skm.average_precision_score(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
      })
    vx, _, _,_,_ =  compute_aupr_df1(gs_df, predict_df, plot=False, percentile=100)
    roc_pr_stats.update(vx)
    if plot is True:
        fpr, tpr, _ = skm.roc_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        precision, recall, _ = skm.precision_recall_curve(combo_df.twt, np.abs(combo_df.pwt.to_numpy()))
        return roc_pr_stats, fpr, tpr, precision, recall
    else:
        return roc_pr_stats, None, None, None, None


def compute_aupr_marbach(gs_df, predict_df):
    ## normalize weights between 0 and 1
    lpredict_df = predict_df.copy()
    lpredict_df['pwt'] = abs(lpredict_df.pwt)
    pred_wt = lpredict_df['pwt']
    norm_pred_wt=(pred_wt-pred_wt.min())/(pred_wt.max()-pred_wt.min())
    lpredict_df['pwt'] = norm_pred_wt
    #
    T = gs_df.shape[0]
    P = np.sum(gs_df.twt)
    N = T - P
    ##
    pdf = predict_df.merge(gs_df, on = ["TF", "TARGET"])
    pdf.fillna(0)
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
    combo_df = gs_df.merge(pdf, how='left', on = ["TF", "TARGET"])
    MSNG = combo_df.pwt.isna().sum()
    print("P:", P, "N:", N, "T:", T, "L:", L, "TPL:", TPL, "MSNG:", MSNG, 
            "p:", p, "pdf:", pdf.shape, "combo:", combo_df.shape)
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
    #rx = compute_aupr_marbach(gs_df, predict_df)
    #roc_pr_stats.update(rx)
    ##
    #tgs_df = gs_df.loc[gs_df.TF != gs_df.TARGET]
    #rx2 = compute_aupr_marbach(tgs_df, predict_df)
    #roc_pr_stats.update(rx2)
    #roc_pr_stats.update({'UNIQ_'+x : y for x,y in rx2.items()})
    vx, fpr, tpr, precision, recall =  compute_aupr_df1(gs_df, predict_df, plot, percentile=100)
    roc_pr_stats.update(vx)
    return roc_pr_stats, fpr, tpr, precision, recall

def matrix_df_covert(network_df, gs_df):
    #print(gs_df)
    tf_lst = list(dict.fromkeys([x for x in gs_df.TF]))
    gene_lst = list(dict.fromkeys([x for x in gs_df.TARGET])) 
    tf_lst = [x for x in tf_lst if x in network_df.index]
    gene_lst = [x for x in gene_lst if x in network_df.index]
    vals = network_df.loc[tf_lst,gene_lst].to_numpy().flatten()
    tfnames = [item for item in tf_lst for i in range(len(gene_lst))]
    genenames = gene_lst * len(tf_lst)

    odf = pd.DataFrame(data={'TF': tfnames, 'TARGET': genenames, 'pwt': vals})
    #odf.to_csv(out_cmp_file, "\t", index=False, header=False)
    return odf


def compute_aupr(gs_file, predict_file, plot):
    gs_df = pd.read_csv(gs_file, sep="\t", names=["TF", "TARGET", "twt"])
    predict_df = pd.read_csv(predict_file, sep="\t", names=["TF", "TARGET", "pwt"] )
    return compute_aupr_df(gs_df, predict_df, plot)


def read_hdf(filename, dtype=np.float64):
    return pd.read_hdf(filename, 'array')

def read_tsv(filename):
    return pd.read_csv(filename, sep="\t")

def tsv_combo_roc(merged_df, gs_df, pfx_entry, plot=False):
    mix_ratios = pfx_entry["mix"]
    method_label = pfx_entry["label"]
    method_id = pfx_entry["id"]
    pt = pfx_entry["percentile"]
    mg_df =  merged_df.copy()
    column_names = []
    mg_df["cmi"] = mg_df["mi"]
    for kx,rc in mix_ratios.items():
        mg_df[kx] = mg_df[kx] * rc
        column_names.append(kx)
    mg_df["xwt"] = mg_df[column_names].sum(axis=1)
    mg_df["pwt"] = mg_df["cmi"]/mg_df["xwt"]
    mg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #predict_df = mg_df[['TF', 'TARGET', 'pwt']]
    predict_df = gs_df.merge(mg_df)
    predict_df = predict_df[['TF', 'TARGET', 'pwt']]
    #print("PRED", predict_df.head())
    apx, fpr, tpr, prec, recall = compute_aupr_df(gs_df, predict_df, plot, pt)
    apx.update({"label": method_label, "id": method_id,
                "mix" : mix_ratios, "percentile": pt})
    #if 'STOUFFER' in pfx_entry:
    #    apx['STOUFFER'] = 'Y'
    #print("\t".join([method_id, method_label] + [str(x) for x in cpd]))
    return apx, fpr, tpr, prec, recall 
 
def tsv_diff_combo_roc(merged_df, gs_df, pfx_entry, plot=False):
    mix_ratios = pfx_entry["mix"]
    method_label = pfx_entry["label"]
    method_id = pfx_entry["id"]
    pt = pfx_entry["percentile"]
    mg_df =  merged_df.copy()
    column_names = []
    mg_df["cmi"] = mg_df["mi"]
    for kx,rc in mix_ratios.items():
        mg_df[kx] = mg_df[kx] * rc
        column_names.append(kx)
    mg_df["xwt"] = mg_df[column_names].sum(axis=1)
    mg_df["pwt"] = mg_df["cmi"] - mg_df["xwt"]
    #predict_df = mg_df[['TF', 'TARGET', 'pwt']]
    predict_df = gs_df.merge(mg_df)[['TF', 'TARGET', 'pwt']]
    #print("PRED", predict_df.head())
    apx, fpr, tpr, prec, recall = compute_aupr_df(gs_df, predict_df, plot, pt)
    apx.update({"label": method_label, "id": 'diff_'+method_id,
        "mix" : mix_ratios, "percentile": pt})
    #if 'STOUFFER' in pfx_entry:
    #    apx['STOUFFER'] = 'Y'
    #print("\t".join([method_id, method_label] + [str(x) for x in cpd]))
    return apx, fpr, tpr, prec, recall 
 
def write_best_combo(merged_df, bc_entry, bc_out_file):
    mix_ratios = bc_entry["mix"]
    method_label = bc_entry["label"]
    combo_id = bc_entry["id"]
    mg_df =  merged_df.copy()
    column_names = []
    mg_df["cmi"] = mg_df["mi"]
    for kx,rc in mix_ratios.items():
        mg_df[kx] = mg_df[kx] * rc
        column_names.append(kx)
    if "diff" in combo_id:
        mg_df["xwt"] = mg_df[column_names].sum(axis=1)
        mg_df["wt"] = mg_df["cmi"] - mg_df["xwt"]
    else:
        mg_df["xwt"] = mg_df[column_names].sum(axis=1)
        mg_df["wt"] = mg_df["cmi"]/mg_df["xwt"]
        mg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mg_df.to_csv(bc_out_file)



def load_predict_file(predict_file, map_df, key_column, df_column):
    if map_df is None:
        odf = pd.read_csv(predict_file, sep="\t")
    else:
        odf = read_tsv(predict_file).rename(columns={
            'TF':'TF_PROBE', 'TARGET':'TARGET_PROBE'})
        odf = odf[['TF_PROBE', 'TARGET_PROBE', df_column]]
        odf = odf.merge(map_df, how='left', left_on='TF_PROBE', right_on='PROBE')
        odf = odf.rename(columns={"ID":"TF_ID"})
        odf = odf.merge(map_df, how='left', left_on='TARGET_PROBE', right_on='PROBE')
        odf = odf.rename(columns={"ID":"TARGET_ID"})
        odf = odf.rename(columns={"TF_ID":"TF", "TARGET_ID":"TARGET"})
    odf = odf[["TF","TARGET", df_column]]
    odf2 = odf.rename(columns={"TF":"TARGET", "TARGET":"TF"})
    odf = pd.concat([odf, odf2])
    odf = odf[odf.TF < odf.TARGET]
    odf.rename(columns={df_column : key_column}, inplace=True)
    #print("LOAD", odf.shape, odf.head())
    return odf[["TF", "TARGET", key_column]]

def tsv_method_roc(gs_df, predict_fxentry, map_df, plot=False):
    method_id = predict_fxentry["id"]
    method_label = predict_fxentry["label"]
    predict_file = predict_fxentry["file"]
    pt = predict_fxentry["percentile"]
    df_column = predict_fxentry["column"]
    prec = None
    recall = None
    predict_df = load_predict_file(predict_file, map_df, method_id, df_column)
    predict_df.rename(columns={method_id : "pwt"}, inplace=True)
    #print("METHOD", predict_df.shape, predict_df.head())
    #predict_df = gs_df.merge(odf)
    apx, fpr, tpr, prec, recall = compute_aupr_df(gs_df, predict_df, plot, pt)
    apx.update({"label": method_label, "id" : method_id,
        "mix": [1.0],  "percentile": pt})
    #print("\t".join(out_lst))
    return apx, fpr, tpr, prec, recall 



def main(input_json):
    with open(input_json) as jfptr:
        jsx = json.load(jfptr)
    data_dir = jsx["DATA_DIR"] + "/"
    gs_file = data_dir + jsx["TRUE_NET"]
    key_tsv_files = {kx: data_dir + jsx[fkx] for kx,fkx in jsx["COMBO_KEYS"].items()}
    #stoufferMat = True if 'STOUFFER' in jsx else False
    plt_colors = jsx['PLOT_COLORS']
    out_format = jsx['OUT_FORMAT']
    out_file = jsx["OUT_FILE"]
    fig_size=ast.literal_eval(jsx['FIG_SIZE'])
    subfig_size=ast.literal_eval(jsx['SUB_FIG_SIZE'])
    ncolors = len(plt_colors)
    if 'PERCENTILE' in jsx:
        pt = int(jsx['PERCENTILE'])
    else: 
        pt = 100 
    if "BEST_COMBO_OUT_FILE" in jsx:
        bc_out_file = jsx["BEST_COMBO_OUT_FILE"]
    else:
        bc_out_file = None
    print("Output format  : ", out_format)
    print("Figure Size    : ", fig_size)
    print("SubFigure Size : ", subfig_size)
    #print("Soutffer Mat   : ", stoufferMat)
    print("Percentile     : ", pt)
    print("Key Mat File   : ", ",".join([str(x) for x in key_tsv_files.items()]))
    #
    gs_df = pd.read_csv(gs_file, sep="\t", names=["TF", "TARGET", "twt"])
    gs_df2 = gs_df.rename(columns={"TF":"TARGET", "TARGET":"TF"})
    gs_df = pd.concat([gs_df, gs_df2])
    gs_df = gs_df[gs_df.TF < gs_df.TARGET]
    print("Loaded Ref Network Total {}, POS {}, NEG{}".format(gs_df.shape, 
            gs_df[gs_df['twt']==1].shape, 
            gs_df[gs_df['twt']==0].shape))
    rev_lst = []
    if "PROBE_MAPPING_FILE" in jsx:
        mapping_file = data_dir + jsx["PROBE_MAPPING_FILE"]
    else:
        mapping_file = None
    map_df = None if mapping_file is None else pd.read_csv(mapping_file, sep="\t")
    for idx, rev_entry in enumerate(jsx["REVNET_FILES"]):
        pfx_entry = {}
        pfx_entry["id"] = rev_entry["id"]
        pfx_entry["label"] = rev_entry["label"]
        pfx_entry["file"] = data_dir + rev_entry["file"]
        pfx_entry["mix"] = [as_float(x) for x in rev_entry["mix"]]
        pfx_entry["percentile"] = pt
        pfx_entry["column"] = rev_entry["column"] if "column" in rev_entry else rev_entry["id"]
        #print(pfx_entry["file"])
        apx, _, _, _, _ = tsv_method_roc(gs_df, pfx_entry, map_df)
        rev_lst.append(apx)
    #print(rev_lst)
    combo_lst = []
    if "COMBO_COLUMNS" in jsx:
        combo_columns = jsx["COMBO_COLUMNS"]
        key_tsv_dfs = [load_predict_file(fx, map_df, kx, combo_columns[kx]) 
                          for kx,fx in key_tsv_files.items()]
    else:
        key_tsv_dfs = [load_predict_file(fx, map_df, kx, kx) 
                          for kx,fx in key_tsv_files.items()]
    merged_df = ft.reduce(lambda x, y: pd.merge(x, y, on = ['TF', 'TARGET'],
                                                how="outer"), key_tsv_dfs)
    merged_df.fillna(0, inplace=True)
    for idx, combo_entry in enumerate(jsx['COMBO_MIXES']):
        pfx_entry = {}
        pfx_entry["id"] = combo_entry["id"]
        pfx_entry["label"] = combo_entry["label"]
        pfx_entry["mix"] = {x:as_float(y) for x,y in combo_entry["mix"].items()}
        pfx_entry["percentile"] = pt
        #if stoufferMat is True:
        #    pfx_entry['STOUFFER'] = 'Y'
        apx, _, _, _, _ = tsv_combo_roc(merged_df, gs_df, pfx_entry)
        combo_lst.append(apx)
        apx, _, _, _, _ = tsv_diff_combo_roc(merged_df, gs_df, pfx_entry)
        combo_lst.append(apx)
    rev_lst.sort(key=lambda x:x['UNIQ_AUPR1'] , reverse=True)
    combo_lst.sort(key=lambda x:x['UNIQ_AUPR1'] , reverse=True)
    #print(combo_lst[:3])
    odf = pd.DataFrame(rev_lst + combo_lst)
    odf.to_csv(out_file, index=False)
    if bc_out_file is not None:
        write_best_combo(merged_df, combo_lst[0], bc_out_file)
    if out_format == "none":
        return 
    plot = True
    matplotlib.style.use('seaborn-muted')
    roc_pr_stats = {}
    for idx, cx in enumerate(combo_lst[:5]):
        _, fpr, tpr, prec, recall = tsv_combo_roc(merged_df, gs_df, cx, True)
        roc_pr_stats[cx["label"]] = [fpr, tpr, prec, recall]
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
    fig.savefig(jsx['ROCPR_OUTPUT_FILE']+out_format)


if __name__ == "__main__":
    #    f1 = sys.argv[1]
    #    f2 = sys.argv[2]
    PROG_DESC = """ ROC/PR of Real Networks"""
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("-f", "--combo_json", default="athaliana_combos_options.json",
                       help="Simulated Input Files")
    ARGS = PARSER.parse_args()
    print("""ARG : JSON File : %s """ % (ARGS.combo_json))
    main(ARGS.combo_json)
 
