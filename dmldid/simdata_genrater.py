import pandas as pd
import numpy as np


def true_delta(X, nonnull_x_cnt=10, base_seed=1):
    Xcolumns_cnt = X.shape[1]
    np.random.seed(seed=base_seed)
    _coef = [
        np.random.normal(0, 5) if i <= nonnull_x_cnt else 0 for i in range(Xcolumns_cnt)
    ]
    if nonnull_x_cnt > 8:
        _i, _ii, _iii, _iiii = np.random.choice(Xcolumns_cnt, 4)
        z = (
            X.dot(_coef)
            + X[f"x{_i}"] * X[f"x{_ii}"] * 1
            + X[f"x{_iii}"] * X[f"x{_iiii}"] * 2
        )
    else:
        z = X.dot(_coef)
    z += np.random.normal(loc=0, scale=1, size=X.shape[0])
    z = 1 / (1 + np.exp(-z))
    z = 0.5 * z * X["x0"]

    return z + np.random.normal(loc=0, scale=0.1, size=X.shape[0])

def potential_y(X, nonnull_x_cnt=10, base_seed=1):
    Xcolumns_cnt = X.shape[1]
    np.random.seed(seed=base_seed)
    _coef = [
        np.random.normal(0, 5) if i <= nonnull_x_cnt else 0 for i in range(Xcolumns_cnt)
    ]
    if nonnull_x_cnt > 8:
        _i, _ii, _iii, _iiii = np.random.choice(Xcolumns_cnt, 4)
        z = (
            X.dot(_coef)
            + X[f"x{_i}"] * X[f"x{_ii}"] * 1
            + X[f"x{_iii}"] * X[f"x{_iiii}"] * 2
        )
    else:
        z = X.dot(_coef)
    z += np.random.normal(loc=0, scale=1, size=X.shape[0])
    z = 1 / (1 + np.exp(-z))
    z = 0.5 * z * X["x0"] +50

    return z + np.random.normal(loc=0, scale=0.1, size=X.shape[0])


def generate_simdata_ro(
    base_seed=1, N=200, Xcolumns_cnt=100, nonnull_x_cnt=10, true_att=3
) -> pd.DataFrame:
    assert Xcolumns_cnt >= nonnull_x_cnt
    df_list = []
    cnt_subgroup = 10
    n_subgroup = int(N / cnt_subgroup)

    for sg in range(cnt_subgroup):
        np.random.seed(seed=base_seed + sg)
        _x = []
        [
            _x.append(
                pd.DataFrame(
                    {f"x{i}": np.random.normal(loc=0, scale=i, size=n_subgroup)}
                )
            )
            for i in range(Xcolumns_cnt)
        ]

        X = pd.concat(_x, axis=1)
        nonnull_x_cols = [f"x{i}" for i in range(nonnull_x_cnt)]

        for i, xcol in enumerate(nonnull_x_cols):
            if i % 2 == 0:
                X[xcol] = -4 + sg + np.random.normal(loc=0, scale=3, size=n_subgroup)
            elif i % 3 == 0:
                X[xcol] =1/(sg + 0.1) + np.random.normal(loc=0, scale=0.2, size=n_subgroup)
            elif i % 5 == 0:
                X[xcol] = (X[xcol] * X[nonnull_x_cols[i - 2]] / 25 + 0.6) ** 2
            elif i % 7 == 0:
                X[xcol] = (X[nonnull_x_cols[i - 2]] * X[xcol] / 25 + 0.6) ** 3
            elif i % 11 == 0:
                X[xcol] = X[xcol] / (1 + np.exp(X[nonnull_x_cols[i - 1]])) + 1
            else:
                X[xcol] = np.exp(X[xcol] / 2)

        _diff = true_delta(X, nonnull_x_cnt=nonnull_x_cnt, base_seed=base_seed)

        baseY = np.random.normal(
            loc=50,
            scale=10,
            size=n_subgroup,
        )
        Y = pd.DataFrame(
            {
                "Y_2020": baseY,
                "Y_2021": baseY + _diff,
                "Y_2022": baseY + _diff * 2,
                "Y_2023": baseY + _diff * 3,
            }
        )
        df = pd.concat([Y, X], axis=1)
        df["latent_group"] = sg
        df["latent_ps"] = np.clip(1 - sg / 10, 0.0001, 1 - 0.1)
        df["D"] = df["latent_ps"].apply(lambda x: np.random.binomial(1, x))
        df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)

    df["att_err"] = np.random.normal(
        loc=0,
        scale=abs(true_att) / 10,
        size=N,
    )
    df["Y_2023"] = df.apply(
        lambda x: x["Y_2023"] + true_att + x["att_err"] if x["D"] > 0 else x["Y_2023"],
        axis=1,
    )
    del df["att_err"]
    return df.reset_index().rename(columns={"index": "unit_id"})


def generate_simdata_rcs(
    base_seed=1, N=200, Xcolumns_cnt=100, nonnull_x_cnt=10, nonnull_d_x_cnt=5, true_att=3, full=None
):
    # TODO
    assert Xcolumns_cnt >= nonnull_x_cnt + nonnull_d_x_cnt
    df_list = []
    cnt_subgroup = 10
    n_subgroup = int(N / cnt_subgroup)

    for sg in range(cnt_subgroup):
        np.random.seed(seed=base_seed + sg)
        _x0 = []
        _x1 = []
        [
            _x0.append(
                pd.DataFrame(
                    {f"x{i}": np.random.normal(loc=0, scale=i, size=n_subgroup)}
                )
            )
            for i in range(Xcolumns_cnt)
        ]
        [
            _x1.append(
                pd.DataFrame(
                    {f"x{i}": np.random.normal(loc=0, scale=i, size=n_subgroup)}
                )
            )
            for i in range(Xcolumns_cnt)
        ]

        X0 = pd.concat(_x0, axis=1)
        X1 = pd.concat(_x0, axis=1)
        nonnull_x_cols = [f"x{i}" for i in range(nonnull_x_cnt)]

        for i, xcol in enumerate(nonnull_x_cols):
            if i % 2 == 0:
                X0[xcol] = sg + np.random.normal(loc=0, scale=3, size=n_subgroup)
                X1[xcol] = X0[xcol] + np.random.normal(loc=1, scale=5, size=n_subgroup)
            elif i % 3 == 0:
                X0[xcol] =1/(sg + 0.1) + np.random.normal(loc=0, scale=0.2, size=n_subgroup)
                X1[xcol] = X0[xcol] * np.random.normal(loc=1.5, scale=0.1, size=n_subgroup)
            elif i % 5 == 0:
                X0[xcol] = (X0[xcol] * X0[nonnull_x_cols[i - 2]] / 25 + 0.6) ** 2
                X1[xcol] = X0[xcol] - np.random.normal(loc=5, scale=1, size=n_subgroup)
            elif i % 7 == 0:
                X0[xcol] = (X0[nonnull_x_cols[i - 2]] * X0[xcol] / 25 + 0.6) ** 3
                X1[xcol] = (X0[nonnull_x_cols[i - 2]] * X1[xcol] / 25 + 0.6) ** 3
            elif i % 11 == 0:
                X0[xcol] = X0[xcol] / (1 + np.exp(X0[nonnull_x_cols[i - 1]])) + 1
                X1[xcol] = X0[xcol] + np.random.normal(loc=-2, scale=1, size=n_subgroup)
            else:
                X0[xcol] = np.exp(X0[xcol] / 2)
                X1[xcol] = X0[xcol] - np.random.normal(loc=1, scale=5, size=n_subgroup)

        y0 = potential_y(X0, nonnull_x_cnt=nonnull_x_cnt, base_seed=base_seed)
        y0 = pd.DataFrame({"Y":y0 })
        y1 = potential_y(X1, nonnull_x_cnt=nonnull_x_cnt, base_seed=base_seed+1)
        y1 = pd.DataFrame({"Y":y1 })

        df0 = pd.concat([y0, X0], axis=1)
        df0["T"] = 0
        df1 = pd.concat([y1, X1], axis=1)
        df1["T"] = 1
        df = pd.concat([df0, df1])
        nonnull_d_x_cols = [f"x{i}" for i in range(nonnull_x_cnt, nonnull_x_cnt+nonnull_d_x_cnt)]
        for i, xcol in enumerate(nonnull_d_x_cols):
            if i % 2 == 0:
                df[xcol] = sg + np.random.normal(loc=0, scale=0.5, size=n_subgroup*2)
            elif i % 3 == 0:
                df[xcol] =2/(sg + 0.1) + np.random.normal(loc=0, scale=0.2, size=n_subgroup*2)
            elif i % 5 == 0:
                df[xcol] = (sg / 25 + 0.6) ** 2
            else:
                df[xcol] = np.exp(sg / 2)

        df["latent_group"] = sg
        df["latent_ps"] = np.clip(1 - sg / 10, 0.0001, 1 - 0.1)
        df["D"] = df["latent_ps"].apply(lambda x: np.random.binomial(1, x))
        df["Y"] = df.apply(lambda x: x["Y"] + true_att if x["D"] + x["T"] >1 else x["Y"], axis=1)
        df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)

    return df.sample(N, random_state=base_seed).reset_index(drop=True)

def generate_simdata_rcs_fixX(
    base_seed=1, N=200, Xcolumns_cnt=100, nonnull_x_cnt=10, true_att=3
) -> pd.DataFrame:
    df = generate_simdata_ro(base_seed=base_seed, N=N, Xcolumns_cnt=Xcolumns_cnt, nonnull_x_cnt=nonnull_x_cnt, true_att=true_att).drop(["Y_2020", "Y_2021"], axis=1)
    df["latent_null_probaility"] = 0.5
    np.random.seed(seed=1)
    df["y_flag"] = df["latent_null_probaility"].apply(lambda x: np.random.binomial(1, x))
    df["Y"] = df.apply(lambda x: x["Y_2023"] if x["y_flag"] >0 else x["Y_2022"], axis=1)
    df["T"] = df.apply(lambda x: 1 if x["y_flag"] >0 else 0, axis=1)
    return df.drop(["Y_2022", "Y_2023", "y_flag", "latent_null_probaility"], axis=1)