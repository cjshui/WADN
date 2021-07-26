import numpy as np
import cvxpy as cp
import torch


def Convex(loss, L2_reg):

    """
    loss: src_number loss
    [loss_1, loss_2, ... loss_src_number]
    """
    src_number = len(loss)
    lam = cp.Variable(src_number)
    prob = cp.Problem(
        cp.Minimize(lam @ loss + L2_reg * cp.norm(lam, 2)), [cp.sum(lam) == 1, lam >= 0]
    )
    # prob.solve()
    prob.solve(solver="SCS")
    lam_optimal = lam.value

    return lam_optimal


def BBSL(C, y_t, y_s):

    """
    C confusion matrix (C defined in the sckit learn should be transpose)
    y_t predicted tar label distribution
    y_s ground truth src label distribution

    """

    n = len(y_s)

    # Define and solve the CVXPY problem.
    alpha = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(alpha @ C - y_t)), [alpha @ y_s == 1, alpha >= 0])
    # prob.solve()
    prob.solve(solver="SCS")
    alpha_opt = alpha.value

    return alpha_opt


def partial_BBSL(C, y_t, y_s, sparse_coef):
    """
    C confusion matrix (C defined in the sckit learn should be transpose)
    y_t predicted tar label distribution
    y_s ground truth src label distribution
    sparse_coef the trade-off coefficient to control the sparsely of alpha_t
    """
    n = len(y_s)
    # Define and solve the CVXPY problem.
    alpha = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(alpha @ C - y_t) + sparse_coef * cp.sum(alpha)),
        [alpha @ y_s == 1, alpha >= 0],
    )
    # prob.solve()
    prob.solve(solver="SCS")
    alpha_opt = alpha.value

    return alpha_opt


def NLLSL(C, y_t, y_s):

    """
    C confusion matrix
    y_t predicted tar label distribution
    y_s ground truth src label distribution

    """

    n = len(y_s)
    # Define and solve the CVXPY problem.
    alpha = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(-1 * y_t @ cp.log(alpha @ C)), [alpha @ y_s == 1, alpha >= 0])
    # prob.solve()
    prob.solve(solver="SCS")
    alpha_opt = alpha.value

    return alpha_opt
