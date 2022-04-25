import argparse
import ast
import json
import sklearn
import sklearn.metrics as skm
import numpy as np
import pandas as pd
import matplotlib
import os.path
matplotlib.use('pdf')
import matplotlib.pyplot as plt
#matplotlib.rcParams['font.sans-serif'] = "Myriad Pro"


def remove_dupe_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    such that only the row with max weight is returned

    Parameters
    ----------
    in_df : Input data frame with columns 'source', 'target', wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name such that
    only the row with max weight is returned
    """
    in_df = in_df.sort_values(by=[wt_attr_name], ascending=False)
    return in_df.drop_duplicates(subset=['source', 'target'], keep='first')


def order_network_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    returns rows such that source < target

    Parameters
    ----------
    in_df : Input data frame with columns 'source', 'target', wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name such that
    source entry < target entry
    """
    if (in_df.source < in_df.target).all():
        return remove_dupe_rows(in_df, wt_attr_name)
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in in_df.to_dict('split')['data']]
    tmp_df = pd.DataFrame(tmp_rcds, columns=['source', 'target', wt_attr_name])
    return remove_dupe_rows(tmp_df, wt_attr_name)

def load_eda_network(eda_file: str, wt_attr_name: str = 'wt',
                     delimiter: str = r'\s+') -> pd.DataFrame:
    """
    Load network from edge attribute file file (.eda).
    Eda file lists the edges in the following format
    The first line has the string "Weight". The edges are listed with the following
    format:
    source (itype) target = weight

    where source and target are node ids, itype is the interaction type and weight is
    the edge weight.

    Example:
    Weight
    244901_at (pp) 244902_at = 0.192777
    244901_at (pp) 244926_s_at = 0.0817807


    Parameters
    ----------
    eda_file : Path to the input .eda file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name

    """
    tmp_df = pd.read_csv(eda_file, sep=delimiter, usecols=[0, 2, 4], skiprows=[0],
                         names=['source', 'target', wt_attr_name])
    return order_network_rows(tmp_df, wt_attr_name)


def load_mat_network(mat_file: str, wt_attr_name: str = 'wt',
                     delimiter: str = " ") -> pd.DataFrame:
    """
    Load network with adjacency matrix;
    Adjacency matrix lists the target wts for each source node:
    source_node1 source_node2 source_node_3 ...
    target_node1 weight11 weight12 weight13 ...
    target_node2 weight21 weight22 weight23 ...
    target_node3 weight31 weight32 weight33 ...
    ...
    ...

    Parameters
    ----------
    adj_file : Path to the input .adj file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name
    """
    mat_df = pd.read_csv(mat_file, sep=delimiter, index_col=0)
    mat_cnames = mat_df.columns
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        'source': np.repeat(mat_cnames, mat_size),
        'target': np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.values.flatten()})
    return net_df[net_df.source < net_df.target]


def load_mat_network_glist(mat_file: str, gene_list, header_opt=None, wt_attr_name: str = 'wt',
                          delimiter: str = ",") -> pd.DataFrame:
    """
    Load network with adjacency matrix with no header
    Adjacency matrix lists the target wts for each source node:
    weight11 weight12 weight13 ...
    weight21 weight22 weight23 ...
    weight31 weight32 weight33 ...
    ...
    ...

    Parameters
    ----------
    adj_file : Path to the input .adj file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name
    """
    mat_df = pd.read_csv(mat_file, header=header_opt,
                         sep=delimiter, names=gene_list)
    print(mat_df.head())
    mat_cnames = gene_list
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        'source': np.repeat(mat_cnames, mat_size),
        'target': np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.values.flatten()})
    return net_df[net_df.source < net_df.target]

def load_tsv_network(eda_file: str, header_opt = None, wt_attr_name: str = 'wt') -> pd.DataFrame:
    """
    Load network from edge attribute file file (.eda).
    Eda file lists the edges in the following format
    The first line has the string "Weight". The edges are listed with the following
    format:
    source (itype) target = weight

    where source and target are node ids, itype is the interaction type and weight is
    the edge weight.

    Example:
    Weight
    244901_at (pp) 244902_at = 0.192777
    244901_at (pp) 244926_s_at = 0.0817807


    Parameters
    ----------
    eda_file : Path to the input .eda file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name

    """
    tmp_df = pd.read_csv(eda_file, sep='\t', header=header_opt, index_col=False,
                         names=['source', 'target', wt_attr_name])
    # print(tmp_df)
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in tmp_df.to_dict('split')['data']]
    xdf = pd.DataFrame(tmp_rcds, columns=['source', 'target', wt_attr_name])
    return xdf.sort_values(by=wt_attr_name, ascending=False)


def read_genes(genes_file):
    with open(genes_file) as ifx:
        rgenes = ifx.readlines()
        return [x.strip() for x in rgenes]


def get_lower_triangle(in_mat):
    ill = np.tril_indices(in_mat.shape[0], -1)
    return in_mat[ill]

def aracne_roc(true_complete_file, arance_file, genes_list):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = arance_file
    rnet = load_tsv_network(filename, header_opt=0, wt_attr_name="pwt")
    # print(filename, rnet.columns)
    # print(rnet.shape)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    # print(rnet.shape)
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    true_a = true.values
    pred_a = pred.values
    #print(max(pred_a), min(pred_a))
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    #print(len(thx), max(thx), min(thx))
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("ARACNE (MI)", arance_file,
          max(pred_a), min(pred_a), max(true_a), min(true_a),
          pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def clr_roc(true_complete_file, clr_fname, genes_list):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = clr_fname
    rnet = load_mat_network_glist(filename, genes_list, header_opt=0,
                                  wt_attr_name="pwt", delimiter="\t")
    # print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt.values
    pred = jnnet.pwt.values
    # print(true[1:5], pred[1:5])
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("CLR (MI)", clr_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def fastggm_roc(true_complete_file, fastggm_fname, genes):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    ngenes = len(genes)
    filename = fastggm_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0,
                       usecols=range(2*ngenes + 1, 3*ngenes + 1))
    true = true_complete.loc[genes, genes]
    # pred = pred.values.flatten()
    # true = true.values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true_a.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred_a.values.flatten()
    #print(max(pred), min(pred))
    #sxgn = str(nsamples) + "X" + str(ngenes)
    pred = 1 - pred
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("FastGGM (COR)", fastggm_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def genenet_roc(true_complete_file, genenet_fname, genes):
    true_complete = np.genfromtxt(true_complete_file,
                                  delimiter=' ',
                                  skip_header=1, usecols=range(1, 2001))
    filename = genenet_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    pred['node1'] = pred['node1'] - 1
    pred['node2'] = pred['node2'] - 1
    true = true_complete[pred['node1'].tolist(), pred['node2'].tolist()]
    pred = pred['pval'].values
    pred = 1 - pred
    pred = pred.flatten()
    true = true.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    print("GeneNet (COR)", genenet_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def genie3_roc(true_complete_file, g3_fname, genes_list):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = g3_fname
    genes_list = sorted(genes_list)
    true_a = true_complete.loc[genes_list, genes_list]
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    pred = pred.sort_values(by=['regulatoryGene', 'targetGene'])
    min_pred = min(pred['weight'].values)
    pred_a = pd.DataFrame(min_pred/2, columns=true_a.columns, index=true_a.index).to_numpy()
    # print(pred_a[~np.eye(pred_a.shape[0], dtype=bool)].shape)
    # print(pred['weight'].values.shape)
    pred_a[~np.eye(pred_a.shape[0], dtype=bool)] = pred['weight'].values # pylint: disable=E1136  # pylint/issues/3139
    true_a = true_a.to_numpy()
    pred_a = np.fmax(np.transpose(pred_a), pred_a)
    true_a = get_lower_triangle(true_a) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("GENIE3 (COR)", g3_fname,
          max(pred_a), min(pred_a), max(true_a), min(true_a),
          pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def grnboost_roc(true_complete_file, grn_fname, genes_list):
    filename = grn_fname
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    rnet = load_tsv_network(filename, wt_attr_name="pwt")
    min_vx = min(rnet.pwt.values)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    jnnet.fillna(min_vx/2.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("GRNBoost (LR)", grn_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def inferlator_roc(true_complete_file, infr_fname, genes_list):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = infr_fname
    pred = pd.read_csv(filename, sep='\t',
                       usecols=['regulator', 'target', 'var.exp.median'])
    true = true_complete.loc[genes_list, genes_list]
    min_pred = min(pred['var.exp.median'].values)
    #pred = pred['var.exp.median'].values
    true_a = true
    pred_a = pd.DataFrame(min_pred/2, columns=true_a.columns, index=true_a.index)
    for _, row in pred.iterrows():
        r_max = max(pred_a.loc[row.regulator, row.target], pred_a.loc[row.target, row.regulator])
        r_max = max(r_max, row['var.exp.median'])
        pred_a.loc[row.regulator, row.target] = r_max
        pred_a.loc[row.target, row.regulator] = r_max
    true_a = get_lower_triangle(true_a.to_numpy()) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a.to_numpy()) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("Inferelator (LR)", infr_fname,
          max(pred_a), min(pred_a), max(true_a), min(true_a),
          pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def mrnet_roc(true_complete_file, mrnet_fname, genes_list):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = mrnet_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    rnet = load_mat_network(filename, wt_attr_name="pwt")
    # print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("mrnet (MI)", mrnet_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def tigress_roc(true_complete_file, tig_fname, genes):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = tig_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    ngenes = len(pred.index.values)
    pred = pred[pred.columns[-ngenes:]]
    pred.columns = pred.index.values
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index)
    true = get_lower_triangle(true.to_numpy()) # true_a.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("TIGRESS (LR)", tig_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def tinge_roc(true_complete_file, tinge_fname, genes_list):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = tinge_fname
    rnet = load_eda_network(filename, wt_attr_name="pwt")
    #print(filename, rnet.columns)
    #min_mi = min(pred['wt'].values)
    #print(min_mi)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    # print(jnnet.shape)
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    true_a = true.values
    pred_a = pred.values
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("TINGe (MI)", tinge_fname,
          max(pred_a), min(pred_a), max(true_a), min(true_a),
          pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def wgcna_roc(true_complete_file, tinge_fname, genes=None):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = tinge_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index) # .values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("WGCNA (COR)", tinge_fname,
          max(pred), min(pred), max(true), min(true),
          pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def pcc_roc(true_complete_file, tinge_fname, genes):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = tinge_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index) # .values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true.values.flatten()
    pred = get_lower_triangle(np.abs(pred.to_numpy())) # pred.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    npauc = sum(1 for x in fpr if x <= 0.05)
    npauc2 = sum(1 for x in fpr if x <= 0.01)
    print("PCC (COR)", tinge_fname,
          max(pred), min(pred), max(true), min(true), pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

ROCPR_METHOD_DICT = {
    'aracne'  : aracne_roc,
    'clr'     : clr_roc,
    'fastggm' : fastggm_roc,
    'genenet' : genenet_roc,
    'genie3'  : genie3_roc,
    'grnboost' : grnboost_roc,
    'arboreto' : grnboost_roc,
    'inferlator' : inferlator_roc,
    'mrnet' : mrnet_roc,
    'tigress' : tigress_roc,
    'tinge' : tinge_roc,
    'wgcna' : wgcna_roc,
    'pcc' :  pcc_roc
}

def plot_method_roc(true_edgesfx, predict_fxentry, net_genes, plot=False):
    plot_color = predict_fxentry["color"]
    method_name = predict_fxentry["method"]
    method_label = predict_fxentry["label"]
    predict_edgesfx = predict_fxentry["file"]
    roc_fn = ROCPR_METHOD_DICT[method_name]
    fpr, tpr, _, _ = roc_fn(true_edgesfx, predict_edgesfx, net_genes)
    if plot is True:
        plt.plot(fpr, tpr, color=plot_color, label=method_label)
    return fpr, tpr

def plot_all_roc2k(true_edgesfx, net_genes, jsx, histogram, plot=False):
    fpr_full = []
    tpr_full = []
    method_full = []
    plt_colors = jsx['PLOT_COLORS']
    for idx, file_entry in enumerate(jsx['REVNET_FILES']):
        pfx_entry = {}
        pfx_entry["color"] = plt_colors[idx % len(plt_colors)]
        pfx_entry["method"] = file_entry["method"]
        pfx_entry["label"] = file_entry["label"]
        pfx_entry["file"] = jsx['DATA_DIR'] +  file_entry["file"]
        fpr, tpr = plot_method_roc(true_edgesfx, pfx_entry, net_genes, plot)
        if histogram is True:
             fpr_full = fpr_full + [x for x in fpr]
             tpr_full = tpr_full + [x for x in tpr]
             method_full = method_full + [file_entry["label"] for _ in fpr]
    if histogram is True:
        mdf = pd.DataFrame({"FPR": fpr_full, "TPR": tpr_full,
                            "METHOD": method_full})
        mdf.to_csv(jsx['AUROC_OUT_FILE'], index=False)
    # diagonal line
    if plot is True:
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

def plot_method_pr(true_edgesfx, predict_fxentry, net_genes, plot=False):
    plot_color = predict_fxentry["color"]
    method_name = predict_fxentry["method"]
    method_label = predict_fxentry["label"]
    predict_edgesfx = predict_fxentry["file"]
    roc_fn = ROCPR_METHOD_DICT[method_name]
    _, _, prec, recall = roc_fn(true_edgesfx, predict_edgesfx, net_genes)
    if plot is True:
        plt.plot(prec, recall, color=plot_color, label=method_label)
    return prec, recall

def plot_all_pr2k(true_edgesfx, net_genes, jsx, histogram, plot=False):
    prec_full = []
    recall_full = []
    method_full = []
    plt_colors = jsx['PLOT_COLORS']
    for idx, file_entry in enumerate(jsx['REVNET_FILES']):
        pfx_entry = {}
        pfx_entry["color"] = plt_colors[idx % len(plt_colors)]
        pfx_entry["method"] = file_entry["method"]
        pfx_entry["label"] = file_entry["label"]
        pfx_entry["file"] = jsx['DATA_DIR'] +  file_entry["file"]
        prec, recall = plot_method_pr(true_edgesfx, pfx_entry,
                                      net_genes, plot)
        if histogram is True:
            prec_full = prec_full + [x for x in prec]
            recall_full = recall_full + [x for x in recall]
            method_full = method_full + [file_entry["label"] for _ in prec]
    if histogram is True:
        mdf = pd.DataFrame({"PRECISION": prec_full, "RECALL": recall_full,
                            "METHOD": method_full})
        mdf.to_csv(jsx["AUPR_OUT_FILE"], index=False)
    # diagonal line
    if plot is True:
        plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='upper right')


def main(out_format, split, sim_json, histogram):
    with open(sim_json) as jfptr:
        jsx = json.load(jfptr)
    true_complete_file = jsx['TRUE_NET']
    net_genes = []
    with open(true_complete_file) as ifx:
        line = ifx.readline().replace('"', '')
        net_genes = line.split()
    print(jsx['REVNET_FILES'])
    missing_files = [(idx, file_entry) for idx, file_entry in enumerate(jsx['REVNET_FILES'])
                      if not os.path.exists(jsx['DATA_DIR'] +  file_entry["file"]) ]
    print("MISSING : ", missing_files)
    if (len(missing_files) > 0):
        print("MISSING FILES")
        return
    if out_format is not "none":
        matplotlib.style.use('seaborn-muted')
        fig_size=ast.literal_eval(jsx['FIG_SIZE'])
        subfig_size=ast.literal_eval(jsx['SUB_FIG_SIZE'])
        print(fig_size, subfig_size)
        if split is True:
            fig = plt.figure(figsize=fig_size)
            # plt.subplot(1, 2, 2)
            plot_all_roc2k(true_complete_file, net_genes, jsx, histogram, True)
            fig.savefig(jsx['ROC_OUTPUT_FILE']+out_format)
            fig = plt.figure(figsize=fig_size)
            plot_all_pr2k(true_complete_file, net_genes, jsx, histogram, True)
            fig.savefig(jsx['PR_OUTPUT_FILE']+out_format)
        else:
            fig = plt.figure(figsize=subfig_size)
            plt.subplot(1, 2, 1)
            plot_all_roc2k(true_complete_file, net_genes, jsx, histogram, True)
            plt.subplot(1, 2, 2)
            plot_all_pr2k(true_complete_file, net_genes, jsx, histogram, True)
            fig.savefig(jsx['ROCPR_OUTPUT_FILE']+out_format)
    else:
        plot_all_roc2k(true_complete_file, net_genes, jsx, histogram, False)
        #plot_all_pr2k(true_complete_file, net_genes, jsx, histogram, False)



if __name__ == "__main__":
    PROG_DESC = """ ROC/PR of Simulated Networks"""
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("-g", "--img", default='none', choices=['png', 'pdf', 'none'],
                        help="""Output image options """)
    PARSER.add_argument("-s", "--split", action='store_true',
                        help="""Should PC and ROC be seprate figures""")
    PARSER.add_argument("-f", "--sim_json", default="./auc_pr_json/sim_options2.json",
                       help="Simulated Input Files")
    PARSER.add_argument("-t", "--histogram", action='store_true',
                        help="""Should generate a histogram file ?""")
    ARGS = PARSER.parse_args()
    print("""
       ARG : Output Image Format : %s
       ARG : Should ROC/PR : %s 
       ARG : JSON File : %s 
       ARG : Histogram : %s""" % (ARGS.img, ARGS.split, ARGS.sim_json, ARGS.histogram))
    main(ARGS.img, ARGS.split, ARGS.sim_json, ARGS.histogram)
