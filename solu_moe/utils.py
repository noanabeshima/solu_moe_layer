import torch
import numpy as np
import plotly.graph_objects as go


def get_scheduler(optimizer, n_steps):
    def lr_lambda(step):
        if step < 0.05 * n_steps:
            return step / (0.05 * n_steps)
        else:
            return 1 - (step - 0.05 * n_steps) / (n_steps - 0.05 * n_steps + 1e-3)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def str_arr_add(*args):
    """
    Casts tensors/nparrays to numpy string arrays and adds all items together,
    casting to string and broadcasting if necessary
    """
    if len(args) == 0:
        return ""
    args = list(args)
    for i, item in enumerate(args):
        if isinstance(item, torch.Tensor):
            item = item.numpy()
        if isinstance(item, np.ndarray):
            args[i] = item.astype(str)
    res = args[0]
    for item in args[1:]:
        res = np.core.defchararray.add(res, item)
    return res


def heatmap(
    arr,
    perm_0=False,
    perm_1=False,
    dim_names=("row", "col"),
    info_0=None,
    info_1=None,
    include_idx=(True, True),
    title=False,
):
    """
    name_0, name_1 : names of dim 0 and dim 1 respectively
    info_0, info_1 : dictionary of string keys to list of strings describing the indices of dim 0 and dim 1 respectively

    include_idx[i] must be true if dim_i_info is False
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    assert isinstance(arr, np.ndarray)
    # assert not (info_0 is False and include_idx[0] is False)
    # assert not (info_1 is False and include_idx[1] is False)

    if title is False:
        if dim_names == ("row", "col"):
            title = f"{arr.shape}"
        else:
            title = f"({dim_names[0]}, {dim_names[1]})"

    perm_0 = np.arange(arr.shape[0]) if perm_0 is False else perm_0
    perm_1 = np.arange(arr.shape[1]) if perm_1 is False else perm_1

    if info_0 is None and include_idx[0] is True:
        info_0 = {}
    if info_1 is None and include_idx[1] is True:
        info_1 = {}

    if info_0 is not None and include_idx[0]:
        info_0[f"{dim_names[0]}"] = np.arange(arr.shape[0])
    if info_1 is not None and include_idx[1]:
        info_1[f"{dim_names[1]}"] = np.arange(arr.shape[1])

    assert info_0 != {}
    assert info_1 != {}

    hovertemplate = ""
    if info_0 is not None:
        hovertemplate += "%{y}"
        info_0 = {k: np.array(v)[perm_0] for k, v in info_0.items()}
        info_0 = str_arr_add(
            *[str_arr_add(k + ": ", v, "<br>") for k, v in info_0.items()]
        )
        info_0 = np.array(info_0).astype(str).tolist()

    if info_1 is not None:
        hovertemplate += "%{x}"
        info_1 = {k: np.array(v)[perm_1] for k, v in info_1.items()}
        info_1 = str_arr_add(
            *[str_arr_add(k + ": ", v, "<br>") for k, v in info_1.items()]
        )
        info_1 = np.array(info_1).astype(str).tolist()
    hovertemplate += "val: %{z:.2f}<extra></extra>"

    layout = go.Layout(yaxis=dict(autorange="reversed"))

    fig = go.Figure(
        data=go.Heatmap(
            z=arr[perm_0][:, perm_1],
            x=info_1,
            y=info_0,
            hovertemplate=hovertemplate,
            colorscale="Viridis",
        ),
        layout=layout,
    )

    fig.update_layout(
        xaxis_title=f"{dim_names[1]} ({arr.shape[1]})",
        yaxis_title=f"{dim_names[0]} ({arr.shape[0]})",
        title=title,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig
