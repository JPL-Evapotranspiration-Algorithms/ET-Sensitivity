from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mstats

def repeat_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(df.values, n, axis=0), columns=df.columns)

def joint_perturbed_run(
        input_df: pd.DataFrame, 
        input_variable: str, 
        output_variable: str, 
        forward_process: Callable,
        perturbation_process: Callable = np.random.multivariate_normal,
        n: int = 1000, 
        perturbation_mean: float = None,
        perturbation_cov: float = None,
        dropna: bool = True) -> pd.DataFrame:
    # calculate standard deviation of the input variable

    n_input = len(input_variable)
    n_output = len(output_variable)

    input_std = np.nanstd(input_df[input_variable],axis=0)

    if all(x == 0 for x in input_std):
        input_std = np.empty(n_input) * np.nan

    # use diagonal (independent) standard deviations of the input variables if not given
    if perturbation_cov is None:
        perturbation_cov = np.diag(input_std)

    if perturbation_mean is None:
        perturbation_mean = np.zeros(n_input)

    # forward process the unperturbed input
    unperturbed_output_df = forward_process(input_df)
    # calculate standard deviation of the output variable
    output_std = np.nanstd(unperturbed_output_df[output_variable],axis=0)

    if all(x == 0 for x in output_std):
        output_std = np.empty(n_output) * np.nan

    # extract output variable from unperturbed output
    unperturbed_output = unperturbed_output_df[output_variable]
    # repeat unperturbed output
    unperturbed_output = repeat_rows(unperturbed_output, n)
    # generate input perturbation
    input_perturbation = perturbation_process(perturbation_mean, perturbation_cov, n*len(input_df))
    input_perturbation_std = input_perturbation / input_std
    # copy input for perturbation
    perturbed_input_df = input_df.copy()
    # repeat input for perturbation
    perturbed_input_df = repeat_rows(perturbed_input_df, n)
    # extract input variable from repeated unperturbed input
    unperturbed_input = perturbed_input_df[input_variable]
    # add perturbation to input
    perturbed_input_df[input_variable] = perturbed_input_df[input_variable] + input_perturbation
    # extract perturbed input
    perturbed_input = perturbed_input_df[input_variable]
    print(np.sum(perturbed_input_df.Rn == 0))
    # forward process the perturbed input
    perturbed_output_df = forward_process(perturbed_input_df)
    # extract output variable from perturbed output
    perturbed_output = perturbed_output_df[output_variable]
    # calculate output perturbation
    output_perturbation = perturbed_output - unperturbed_output
    output_perturbation_std = output_perturbation / output_std

    input_perturbation_df = pd.DataFrame(input_perturbation, columns=[s+"_perturbation" for s in input_variable])
    input_perturbation_std_df = pd.DataFrame(input_perturbation_std, columns=[s+"_perturbation_std" for s in input_variable])

    unperturbed_output = unperturbed_output.loc[:,~unperturbed_output.columns.duplicated()]

    unperturbed_input.columns = [s+"_unperturbed" for s in input_variable]
    unperturbed_output.columns = [s+"_unperturbed" for s in output_variable]
    perturbed_input.columns = [s+"_perturbed" for s in input_variable]
    output_perturbation.columns = [s+"_perturbation" for s in output_variable]
    output_perturbation_std.columns = [s+"_perturbation_std" for s in output_variable]
    perturbed_output.columns = [s+"_perturbed" for s in output_variable]

    ids = pd.DataFrame({"id": np.repeat(np.arange(input_df.shape[0]),n)})

    results_df = pd.concat([unperturbed_input,
                            input_perturbation_df,
                            input_perturbation_std_df,
                            perturbed_input,
                            unperturbed_output,
                            output_perturbation,
                            output_perturbation_std,
                            perturbed_output,
                            ids], axis=1)

    if dropna:
        results_df = results_df.dropna()

    return unperturbed_input, perturbed_input_df, perturbed_output_df, results_df
