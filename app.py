# ============================================================
# OLD FAITHFUL: TWO-NORMAL MIXTURE MODEL
# ============================================================
#
# Features:
# 1. Old Faithful dataset exploration
# 2. Interactive two-normal mixture density
# 3. Coin-toss likelihood and MLE
# 4. Maximum likelihood fitting of a two-normal mixture
# 5. Tail probability calculation
# 6. Numerical Hessian and eigenvalue check
#
# Required packages:
# streamlit, numpy, pandas, matplotlib, scipy, statsmodels
# ============================================================


import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import minimize
from scipy.special import logsumexp, xlog1py, xlogy
from scipy.stats import norm

from statsmodels.datasets import get_rdataset


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Old Faithful | Normal Mixture",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background-color: #f6f8fc;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #14213d;
    }

    section[data-testid="stSidebar"] * {
        color: #f4f7ff;
    }

    /* Sidebar input backgrounds */
    section[data-testid="stSidebar"] div[data-baseweb="select"] *,
    section[data-testid="stSidebar"] input {
        color: #172033 !important;
    }

    /* Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #14213d;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #53627c;
        margin-bottom: 1.8rem;
    }

    /* Information card */
    .info-card {
        background: white;
        border: 1px solid #e3e9f3;
        border-left: 5px solid #4361ee;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.8rem 0 1.2rem 0;
        color: #25324b;
    }

    /* Small tag */
    .small-tag {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        background-color: #e8eeff;
        color: #3451c6;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e9f3;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(20, 33, 61, 0.04);
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: none;
        padding: 0.65rem 1.1rem;
    }

    div.stButton > button[kind="primary"] {
        background-color: #4361ee;
        color: white;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d8e0ed",
        "axes.labelcolor": "#23324d",
        "xtick.color": "#53627c",
        "ytick.color": "#53627c",
        "text.color": "#23324d",
        "axes.titleweight": "bold",
        "font.size": 10,
        "grid.color": "#dce4ef",
        "grid.alpha": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_faithful_data():
    """
    Load the R datasets::faithful dataset using statsmodels.

    The first download requires an internet connection.
    After downloading, statsmodels can use its local cache.
    """

    dataset = get_rdataset(
        dataname="faithful",
        package="datasets",
        cache=True,
    )

    data = dataset.data.copy()

    return data[["eruptions", "waiting"]]


# ============================================================
# NORMAL MIXTURE FUNCTIONS
# ============================================================

def normal_mixture_pdf(x, mu_1, sigma_1, mu_2, sigma_2, p):
    """
    Probability density function of a two-normal mixture.
    """

    density_1 = norm.pdf(
        x,
        loc=mu_1,
        scale=sigma_1,
    )

    density_2 = norm.pdf(
        x,
        loc=mu_2,
        scale=sigma_2,
    )

    mixture = (
        p * density_1
        +
        (1.0 - p) * density_2
    )

    return mixture


def normal_mixture_nll(theta, data):
    """
    Negative log-likelihood for the two-normal mixture.

    Parameter order:
        theta = [mu_1, sigma_1, mu_2, sigma_2, p]

    logsumexp improves numerical stability.
    """

    mu_1, sigma_1, mu_2, sigma_2, p = theta

    # These checks protect the logarithms and normal densities.
    if sigma_1 <= 0 or sigma_2 <= 0:
        return np.inf

    if p <= 0 or p >= 1:
        return np.inf

    log_component_1 = (
        np.log(p)
        +
        norm.logpdf(
            data,
            loc=mu_1,
            scale=sigma_1,
        )
    )

    log_component_2 = (
        np.log1p(-p)
        +
        norm.logpdf(
            data,
            loc=mu_2,
            scale=sigma_2,
        )
    )

    log_density = logsumexp(
        np.vstack(
            [
                log_component_1,
                log_component_2,
            ]
        ),
        axis=0,
    )

    if not np.all(np.isfinite(log_density)):
        return np.inf

    return float(-np.sum(log_density))


def normal_mixture_tail_probability(
    threshold,
    mu_1,
    sigma_1,
    mu_2,
    sigma_2,
    p,
):
    """
    Calculate P(X > threshold) for a two-normal mixture.
    """

    tail_1 = norm.sf(
        threshold,
        loc=mu_1,
        scale=sigma_1,
    )

    tail_2 = norm.sf(
        threshold,
        loc=mu_2,
        scale=sigma_2,
    )

    return (
        p * tail_1
        +
        (1.0 - p) * tail_2
    )


def normal_mixture_interval_probability(
    lower,
    upper,
    mu_1,
    sigma_1,
    mu_2,
    sigma_2,
    p,
):
    """
    Calculate P(lower < X < upper) for a two-normal mixture.
    """

    probability_1 = (
        norm.cdf(upper, loc=mu_1, scale=sigma_1)
        -
        norm.cdf(lower, loc=mu_1, scale=sigma_1)
    )

    probability_2 = (
        norm.cdf(upper, loc=mu_2, scale=sigma_2)
        -
        norm.cdf(lower, loc=mu_2, scale=sigma_2)
    )

    return (
        p * probability_1
        +
        (1.0 - p) * probability_2
    )


# ============================================================
# NUMERICAL HESSIAN
# ============================================================

def numerical_hessian(function, x, args=()):
    """
    Approximate the Hessian using central finite differences.

    The Hessian is evaluated in the original parameter scale.
    The result is meaningful only when the point is sufficiently
    far from optimization boundaries.
    """

    x = np.asarray(x, dtype=float)

    n_parameters = len(x)

    hessian = np.zeros(
        (n_parameters, n_parameters),
        dtype=float,
    )

    # Parameter-specific step sizes.
    steps = 1e-4 * np.maximum(
        1.0,
        np.abs(x),
    )

    f_at_x = function(x, *args)

    for i in range(n_parameters):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += steps[i]
        x_minus[i] -= steps[i]

        f_plus = function(x_plus, *args)
        f_minus = function(x_minus, *args)

        hessian[i, i] = (
            f_plus
            -
            2.0 * f_at_x
            +
            f_minus
        ) / (steps[i] ** 2)

        for j in range(i + 1, n_parameters):

            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += steps[i]
            x_pp[j] += steps[j]

            x_pm[i] += steps[i]
            x_pm[j] -= steps[j]

            x_mp[i] -= steps[i]
            x_mp[j] += steps[j]

            x_mm[i] -= steps[i]
            x_mm[j] -= steps[j]

            f_pp = function(x_pp, *args)
            f_pm = function(x_pm, *args)
            f_mp = function(x_mp, *args)
            f_mm = function(x_mm, *args)

            value = (
                f_pp
                -
                f_pm
                -
                f_mp
                +
                f_mm
            ) / (4.0 * steps[i] * steps[j])

            hessian[i, j] = value
            hessian[j, i] = value

    # Remove small numerical asymmetry.
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


# ============================================================
# MAXIMUM LIKELIHOOD FITTING
# ============================================================

@st.cache_data
def fit_normal_mixture(data_tuple):
    """
    Fit a two-normal mixture using bounded maximum likelihood.

    Several starting points are used because mixture models
    can have multiple local optima.

    Bounds prevent zero standard deviations and mixing
    proportions equal to zero or one.
    """

    data = np.asarray(
        data_tuple,
        dtype=float,
    )

    if len(data) < 5:
        raise ValueError(
            "At least five observations are required."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "The waiting-time data contain invalid values."
        )

    sample_mean = float(np.mean(data))
    sample_sd = float(np.std(data, ddof=1))

    if sample_sd <= 0:
        raise ValueError(
            "The data must have positive sample variability."
        )

    q25, q50, q75 = np.quantile(
        data,
        [0.25, 0.50, 0.75],
    )

    # Bounds are chosen using the observed data.
    # They are part of the fitted model specification.
    mean_lower = float(np.min(data) - 2.0 * sample_sd)
    mean_upper = float(np.max(data) + 2.0 * sample_sd)

    sd_lower = max(
        0.1,
        0.02 * sample_sd,
    )

    sd_upper = 3.0 * sample_sd

    bounds = [
        (mean_lower, mean_upper),  # mu_1
        (sd_lower, sd_upper),      # sigma_1
        (mean_lower, mean_upper),  # mu_2
        (sd_lower, sd_upper),      # sigma_2
        (0.001, 0.999),            # p
    ]

    initial_sd = max(
        sample_sd / 2.0,
        sd_lower * 2.0,
    )

    initial_sd = min(
        initial_sd,
        sd_upper * 0.8,
    )

    starting_points = [
        [q25, initial_sd, q75, initial_sd, 0.35],

        [
            q25,
            max(initial_sd * 0.7, sd_lower * 2),
            q75,
            max(initial_sd * 0.8, sd_lower * 2),
            0.50,
        ],

        [
            sample_mean - 0.7 * sample_sd,
            initial_sd,
            sample_mean + 0.7 * sample_sd,
            initial_sd,
            0.40,
        ],

        [
            q50 - 0.5 * sample_sd,
            initial_sd,
            q75,
            initial_sd,
            0.60,
        ],

        [50.0, 6.0, 80.0, 8.0, 0.35],
    ]

    results = []

    for initial in starting_points:

        # Clip initial values to the permitted bounds.
        initial = np.array(
            [
                np.clip(value, lower, upper)
                for value, (lower, upper)
                in zip(initial, bounds)
            ],
            dtype=float,
        )

        result = minimize(
            fun=normal_mixture_nll,
            x0=initial,
            args=(data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )

        if np.isfinite(result.fun):
            results.append(result)

    if len(results) == 0:
        raise RuntimeError(
            "The optimizer could not find a finite likelihood."
        )

    successful_results = [
        result
        for result in results
        if result.success
    ]

    # Choose the best converged solution when available.
    # If none converged, retain the best finite result and warn.
    if successful_results:
        candidate_results = successful_results
    else:
        candidate_results = results

    best_result = min(
        candidate_results,
        key=lambda result: result.fun,
    )

    theta = np.array(
        best_result.x,
        dtype=float,
    )

    # Label component 1 as the component with the lower mean.
    # Swapping the components does not change the mixture.
    if theta[0] > theta[2]:

        theta = np.array(
            [
                theta[2],
                theta[3],
                theta[0],
                theta[1],
                1.0 - theta[4],
            ],
            dtype=float,
        )

    return {
        "theta": theta,
        "nll": float(
            normal_mixture_nll(theta, data)
        ),
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "bounds": bounds,
        "number_of_starts": len(starting_points),
        "number_converged": len(successful_results),
    }


# ============================================================
# COIN-TOSS LIKELIHOOD
# ============================================================

def coin_log_likelihood(p, heads, tails):
    """
    Log-likelihood for independent Bernoulli observations.

    The binomial coefficient is omitted because it does not
    affect the location of the maximum.
    """

    return (
        xlogy(heads, p)
        +
        xlog1py(tails, -p)
    )


# ============================================================
# GRAPH HELPER
# ============================================================

def show_figure(fig):
    """
    Display and close a Matplotlib figure.
    """

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# LOAD DATA
# ============================================================

try:

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        faithful = load_faithful_data()

except Exception as error:

    st.error(
        "The Old Faithful dataset could not be loaded."
    )

    st.info(
        """
        The first download requires an internet connection.

        Check your internet connection and restart the app.
        """
    )

    with st.expander("Show technical details"):
        st.code(str(error))

    st.stop()


waiting = faithful["waiting"].to_numpy(
    dtype=float
)

eruptions = faithful["eruptions"].to_numpy(
    dtype=float
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌋 Old Faithful")

    st.markdown(
        "### Normal Mixture Explorer"
    )

    st.markdown("---")

    st.markdown(
        "**Choose a section**"
    )

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Dashboard",
            "📊 Explore Data",
            "🎨 Build a Mixture",
            "🪙 Coin-Toss Likelihood",
            "🧠 Fit the Model",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        "**Dataset information**"
    )

    st.write(
        f"Observations: **{len(faithful)}**"
    )

    st.write(
        "Variables: **2**"
    )

    st.write(
        "Waiting time: **minutes**"
    )

    st.markdown("---")

    st.caption(
        """
        An interactive app for learning probability,
        likelihood, and normal mixture models.
        """
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="small-tag">'
    'INTERACTIVE STATISTICS APPLICATION'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">'
    '🌋 Old Faithful Explorer'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Explore waiting times, probability densities, '
    'and maximum likelihood estimation.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "Welcome to the Dashboard"
    )

    st.markdown(
        """
        <div class="info-card">
        This app explores the Old Faithful geyser data and shows how
        a mixture of two normal distributions can describe waiting
        times between eruptions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_1, col_2, col_3, col_4 = st.columns(4)

    col_1.metric(
        "Observations",
        f"{len(waiting)}",
    )

    col_2.metric(
        "Average waiting time",
        f"{np.mean(waiting):.2f} min",
    )

    col_3.metric(
        "Median waiting time",
        f"{np.median(waiting):.2f} min",
    )

    col_4.metric(
        "Standard deviation",
        f"{np.std(waiting, ddof=1):.2f} min",
    )

    st.write("")

    left, right = st.columns(
        [1.2, 1.0]
    )

    with left:

        st.subheader(
            "Waiting-Time Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            waiting,
            bins=15,
            density=True,
            color="#5b7cfa",
            edgecolor="white",
            alpha=0.85,
        )

        ax.set_xlabel(
            "Waiting time to next eruption (minutes)"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            "Old Faithful waiting times"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        show_figure(fig)

    with right:

        st.subheader(
            "What can you do?"
        )

        st.markdown(
            """
            **📊 Explore Data**

            View the data, descriptive statistics, scatter plot,
            and histogram.

            **🎨 Build a Mixture**

            Change the means, standard deviations, and mixing
            proportion to see their effect.

            **🪙 Coin-Toss Likelihood**

            Learn how likelihood identifies the maximum likelihood
            estimate of a probability.

            **🧠 Fit the Model**

            Estimate the two-normal mixture parameters and calculate
            waiting-time probabilities.
            """
        )

    st.subheader(
        "The two-normal mixture model"
    )

    st.latex(
        r"""
        f(x)
        =
        p\phi(x;\mu_1,\sigma_1)
        +
        (1-p)\phi(x;\mu_2,\sigma_2)
        """
    )

    st.write(
        """
        Here, \(p\) is the proportion assigned to the first normal
        component. The remaining proportion, \(1-p\), belongs to the
        second component.
        """
    )


# ============================================================
# PAGE 2: EXPLORE DATA
# ============================================================

elif page == "📊 Explore Data":

    st.header(
        "Explore the Old Faithful Data"
    )

    st.markdown(
        """
        <div class="info-card">
        The dataset contains eruption duration and waiting time until
        the next eruption. Both variables are measured in minutes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Dataset preview"
    )

    show_all = st.checkbox(
        "Show all observations",
        value=False,
    )

    if show_all:

        st.dataframe(
            faithful,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.dataframe(
            faithful.head(10),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Waiting-time statistics"
    )

    col_1, col_2, col_3, col_4 = st.columns(4)

    col_1.metric(
        "Mean",
        f"{np.mean(waiting):.2f}",
    )

    col_2.metric(
        "Variance",
        f"{np.var(waiting, ddof=1):.2f}",
    )

    col_3.metric(
        "Standard deviation",
        f"{np.std(waiting, ddof=1):.2f}",
    )

    col_4.metric(
        "Median",
        f"{np.median(waiting):.2f}",
    )

    st.caption(
        "Variance and standard deviation use n − 1, "
        "as in R's var() and sd()."
    )

    with st.expander(
        "View detailed summary statistics"
    ):

        summary = faithful.describe().T

        st.dataframe(
            summary.style.format("{:.2f}"),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader(
        "Visualize the data"
    )

    graph_choice = st.radio(
        "Choose graph",
        options=[
            "Histogram",
            "Scatter plot",
            "Box plot",
        ],
        horizontal=True,
    )

    if graph_choice == "Histogram":

        col_1, col_2 = st.columns(2)

        with col_1:

            bins = st.slider(
                "Number of bins",
                min_value=5,
                max_value=40,
                value=15,
                step=1,
            )

        with col_2:

            histogram_scale = st.selectbox(
                "Histogram scale",
                options=[
                    "Density",
                    "Frequency",
                ],
            )

        density_option = (
            histogram_scale == "Density"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            waiting,
            bins=bins,
            density=density_option,
            color="#5b7cfa",
            edgecolor="white",
            alpha=0.88,
        )

        ax.set_xlabel(
            "Waiting time (minutes)"
        )

        ax.set_ylabel(
            "Density"
            if density_option
            else "Frequency"
        )

        ax.set_title(
            "Histogram of waiting times"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        show_figure(fig)

        if density_option:

            st.info(
                """
                In a density histogram, the total area of the bars is
                1. A bar height is a density, not a probability.
                """
            )

    elif graph_choice == "Scatter plot":

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            eruptions,
            waiting,
            color="#4361ee",
            s=45,
            alpha=0.70,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel(
            "Eruption duration (minutes)"
        )

        ax.set_ylabel(
            "Waiting time (minutes)"
        )

        ax.set_title(
            "Eruption duration and waiting time"
        )

        ax.grid(alpha=0.3)

        show_figure(fig)

    else:

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.boxplot(
            waiting,
            vert=False,
            patch_artist=True,
            boxprops={
                "facecolor": "#95a9ff",
                "edgecolor": "#4361ee",
            },
            medianprops={
                "color": "#d62828",
                "linewidth": 2,
            },
            whiskerprops={
                "color": "#4361ee",
            },
            capprops={
                "color": "#4361ee",
            },
        )

        ax.set_xlabel(
            "Waiting time (minutes)"
        )

        ax.set_yticks([])

        ax.set_title(
            "Box plot of waiting times"
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        show_figure(fig)


# ============================================================
# PAGE 3: BUILD A MIXTURE
# ============================================================

elif page == "🎨 Build a Mixture":

    st.header(
        "Build Your Own Normal Mixture"
    )

    st.markdown(
        """
        <div class="info-card">
        Use the sidebar controls to change the two normal components.
        The graph updates automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        f(x)
        =
        p\phi(x;\mu_1,\sigma_1)
        +
        (1-p)\phi(x;\mu_2,\sigma_2)
        """
    )

    with st.sidebar:

        st.markdown("---")

        st.subheader(
            "🎛️ Mixture controls"
        )

        p_manual = st.slider(
            "Mixing proportion p",
            min_value=0.01,
            max_value=0.99,
            value=0.35,
            step=0.01,
        )

        st.markdown(
            "**Component 1**"
        )

        mu_1_manual = st.slider(
            "Mean μ₁",
            min_value=20.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
        )

        sigma_1_manual = st.slider(
            "Standard deviation σ₁",
            min_value=1.0,
            max_value=25.0,
            value=6.0,
            step=0.5,
        )

        st.markdown(
            "**Component 2**"
        )

        mu_2_manual = st.slider(
            "Mean μ₂",
            min_value=20.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        sigma_2_manual = st.slider(
            "Standard deviation σ₂",
            min_value=1.0,
            max_value=25.0,
            value=8.0,
            step=0.5,
        )

        show_histogram = st.checkbox(
            "Show data histogram",
            value=True,
        )

        show_components = st.checkbox(
            "Show individual components",
            value=True,
        )

    col_1, col_2, col_3 = st.columns(3)

    col_1.metric(
        "Component 1 proportion",
        f"{p_manual:.2f}",
    )

    col_2.metric(
        "Component 2 proportion",
        f"{1.0 - p_manual:.2f}",
    )

    mixture_mean = (
        p_manual * mu_1_manual
        +
        (1.0 - p_manual) * mu_2_manual
    )

    col_3.metric(
        "Mixture mean",
        f"{mixture_mean:.2f} min",
    )

    x_min = min(
        0.0,
        mu_1_manual - 4 * sigma_1_manual,
        mu_2_manual - 4 * sigma_2_manual,
    )

    x_max = max(
        120.0,
        mu_1_manual + 4 * sigma_1_manual,
        mu_2_manual + 4 * sigma_2_manual,
    )

    x_grid = np.linspace(
        x_min,
        x_max,
        1200,
    )

    component_1 = (
        p_manual
        *
        norm.pdf(
            x_grid,
            loc=mu_1_manual,
            scale=sigma_1_manual,
        )
    )

    component_2 = (
        (1.0 - p_manual)
        *
        norm.pdf(
            x_grid,
            loc=mu_2_manual,
            scale=sigma_2_manual,
        )
    )

    mixture_density = (
        component_1 + component_2
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    if show_histogram:

        ax.hist(
            waiting,
            bins=15,
            density=True,
            color="#d7deeb",
            edgecolor="white",
            alpha=0.80,
            label="Observed waiting times",
        )

    if show_components:

        ax.plot(
            x_grid,
            component_1,
            color="#00a896",
            linestyle="--",
            linewidth=2,
            label="Weighted component 1",
        )

        ax.plot(
            x_grid,
            component_2,
            color="#ff9f1c",
            linestyle="--",
            linewidth=2,
            label="Weighted component 2",
        )

    ax.plot(
        x_grid,
        mixture_density,
        color="#4361ee",
        linewidth=3,
        label="Normal mixture",
    )

    ax.fill_between(
        x_grid,
        mixture_density,
        color="#4361ee",
        alpha=0.10,
    )

    ax.set_xlabel(
        "Waiting time (minutes)"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        "Interactive two-normal mixture"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    show_figure(fig)

    st.caption(
        """
        A normal distribution allows negative values, although waiting
        time cannot be negative. Check the fitted model's probability
        below zero when interpreting it.
        """
    )


# ============================================================
# PAGE 4: COIN-TOSS LIKELIHOOD
# ============================================================

elif page == "🪙 Coin-Toss Likelihood":

    st.header(
        "Coin-Toss Likelihood"
    )

    st.markdown(
        """
        <div class="info-card">
        A likelihood function shows how well different values of p
        explain the observed coin-toss results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        L(p)
        =
        p^m(1-p)^{n-m}
        """
    )

    st.write(
        """
        Here, \(n\) is the number of tosses and \(m\) is the number
        of heads.
        """
    )

    with st.sidebar:

        st.markdown("---")

        st.subheader(
            "🪙 Coin-toss controls"
        )

        n_tosses = st.number_input(
            "Number of tosses, n",
            min_value=1,
            max_value=1000,
            value=6,
            step=1,
        )

        n_heads = st.number_input(
            "Number of heads, m",
            min_value=0,
            max_value=int(n_tosses),
            value=min(5, int(n_tosses)),
            step=1,
        )

    n_tosses = int(n_tosses)

    n_heads = int(n_heads)

    n_tails = (
        n_tosses - n_heads
    )

    p_hat = (
        n_heads / n_tosses
    )

    col_1, col_2, col_3 = st.columns(3)

    col_1.metric(
        "Number of tosses",
        n_tosses,
    )

    col_2.metric(
        "Number of heads",
        n_heads,
    )

    col_3.metric(
        "MLE of p",
        f"{p_hat:.4f}",
    )

    st.latex(
        r"""
        \widehat{p}
        =
        \frac{m}{n}
        """
    )

    p_grid = np.linspace(
        0.0,
        1.0,
        1001,
    )

    log_likelihood = coin_log_likelihood(
        p_grid,
        heads=n_heads,
        tails=n_tails,
    )

    # Relative likelihood remains readable even for large n.
    finite_values = log_likelihood[
        np.isfinite(log_likelihood)
    ]

    max_log_likelihood = np.max(
        finite_values
    )

    relative_likelihood = np.exp(
        log_likelihood - max_log_likelihood
    )

    negative_log_likelihood = (
        -log_likelihood
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    axes[0].plot(
        p_grid,
        relative_likelihood,
        color="#4361ee",
        linewidth=2.5,
    )

    axes[0].axvline(
        p_hat,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label=f"MLE = {p_hat:.3f}",
    )

    axes[0].set_xlabel(
        "Probability of heads, p"
    )

    axes[0].set_ylabel(
        "Relative likelihood"
    )

    axes[0].set_title(
        "Relative likelihood"
    )

    axes[0].legend()

    axes[0].grid(alpha=0.3)

    axes[1].plot(
        p_grid,
        negative_log_likelihood,
        color="#00a896",
        linewidth=2.5,
    )

    axes[1].axvline(
        p_hat,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label=f"MLE = {p_hat:.3f}",
    )

    axes[1].set_xlabel(
        "Probability of heads, p"
    )

    axes[1].set_ylabel(
        "Negative log-likelihood"
    )

    axes[1].set_title(
        "Negative log-likelihood"
    )

    axes[1].legend()

    axes[1].grid(alpha=0.3)

    show_figure(fig)

    st.caption(
        """
        Relative likelihood is the likelihood divided by its maximum.
        It has the same maximizer and is easier to display for large
        samples.
        """
    )

    if n_heads == 0 or n_heads == n_tosses:

        st.info(
            """
            The maximum occurs at the boundary because the observed
            outcomes are all tails or all heads.
            """
        )


# ============================================================
# PAGE 5: FIT THE NORMAL MIXTURE
# ============================================================

elif page == "🧠 Fit the Model":

    st.header(
        "Fit the Two-Normal Mixture"
    )

    st.markdown(
        """
        <div class="info-card">
        Estimate the model parameters using maximum likelihood.
        The optimizer tries several starting points to reduce the
        chance of relying on one local solution.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \theta
        =
        (\mu_1,\sigma_1,\mu_2,\sigma_2,p)
        """
    )

    st.latex(
        r"""
        -\ell(\theta)
        =
        -\sum_{i=1}^{n}
        \log
        \left[
        p\phi(x_i;\mu_1,\sigma_1)
        +
        (1-p)\phi(x_i;\mu_2,\sigma_2)
        \right]
        """
    )

    button_col_1, button_col_2, _ = st.columns(
        [1.4, 1.0, 3.0]
    )

    with button_col_1:

        fit_clicked = st.button(
            "🚀 Fit Normal Mixture",
            type="primary",
            use_container_width=True,
        )

    with button_col_2:

        clear_clicked = st.button(
            "🗑️ Clear Results",
            use_container_width=True,
        )

    if clear_clicked:

        if "normal_mixture_fit" in st.session_state:
            del st.session_state["normal_mixture_fit"]

        st.rerun()

    if fit_clicked:

        try:

            with st.spinner(
                "Estimating model parameters..."
            ):

                fit_result = fit_normal_mixture(
                    tuple(waiting.tolist())
                )

            st.session_state[
                "normal_mixture_fit"
            ] = fit_result

        except Exception as error:

            st.error(
                "Model fitting failed."
            )

            with st.expander(
                "Show technical details"
            ):
                st.code(str(error))

    if "normal_mixture_fit" not in st.session_state:

        st.info(
            """
            Click **Fit Normal Mixture** to estimate the parameters.
            """
        )

    else:

        fit_result = st.session_state[
            "normal_mixture_fit"
        ]

        theta_hat = np.asarray(
            fit_result["theta"],
            dtype=float,
        )

        mu_1_hat = float(theta_hat[0])

        sigma_1_hat = float(theta_hat[1])

        mu_2_hat = float(theta_hat[2])

        sigma_2_hat = float(theta_hat[3])

        p_hat = float(theta_hat[4])

        bounds = fit_result["bounds"]

        if fit_result["success"]:

            st.success(
                "The optimizer reported convergence."
            )

        else:

            st.warning(
                """
                A finite result was found, but the optimizer did not
                report convergence. Interpret the estimates cautiously.
                """
            )

        st.caption(
            f"Optimizer message: {fit_result['message']}"
        )

        st.caption(
            f"Converged starting points: "
            f"{fit_result['number_converged']} of "
            f"{fit_result['number_of_starts']}"
        )

        st.subheader(
            "Estimated parameters"
        )

        parameter_names = [
            "μ₁: mean of component 1",
            "σ₁: standard deviation of component 1",
            "μ₂: mean of component 2",
            "σ₂: standard deviation of component 2",
            "p: proportion of component 1",
        ]

        parameter_table = pd.DataFrame(
            {
                "Parameter": parameter_names,
                "Estimate": theta_hat,
            }
        )

        st.dataframe(
            parameter_table.style.format(
                {"Estimate": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            """
            Component 1 is labelled as the component with the smaller
            estimated mean. The labels can be exchanged without
            changing the mixture density.
            """
        )

        # Check if the estimates are close to the bounds.
        near_boundary = []

        for i, (lower, upper) in enumerate(bounds):

            distance_to_bound = min(
                abs(theta_hat[i] - lower),
                abs(theta_hat[i] - upper),
            )

            if distance_to_bound < (
                0.01 * (upper - lower)
            ):

                near_boundary.append(
                    parameter_names[i]
                )

        if near_boundary:

            st.warning(
                "The following estimates are close to an optimization "
                "boundary: "
                +
                ", ".join(near_boundary)
                +
                ". Results may depend on the chosen bounds."
            )

        nll = fit_result["nll"]

        log_likelihood = -nll

        n_parameters = 5

        n_observations = len(waiting)

        aic = (
            2 * n_parameters
            -
            2 * log_likelihood
        )

        bic = (
            np.log(n_observations) * n_parameters
            -
            2 * log_likelihood
        )

        col_1, col_2, col_3 = st.columns(3)

        col_1.metric(
            "Log-likelihood",
            f"{log_likelihood:.2f}",
        )

        col_2.metric(
            "AIC",
            f"{aic:.2f}",
        )

        col_3.metric(
            "BIC",
            f"{bic:.2f}",
        )

        st.caption(
            """
            AIC and BIC are useful for comparing models fitted to the
            same dataset. A smaller value is preferred within the
            corresponding criterion.
            """
        )

        st.markdown("---")

        st.subheader(
            "Fitted density and probability"
        )

        with st.sidebar:

            st.markdown("---")

            st.subheader(
                "📍 Probability controls"
            )

            threshold = st.slider(
                "Waiting-time threshold",
                min_value=0,
                max_value=120,
                value=80,
                step=1,
            )

            show_fitted_components = st.checkbox(
                "Show fitted components",
                value=True,
            )

            show_tail_area = st.checkbox(
                "Shade area above threshold",
                value=True,
            )

        plot_min = min(
            0.0,
            mu_1_hat - 4 * sigma_1_hat,
            mu_2_hat - 4 * sigma_2_hat,
        )

        plot_max = max(
            120.0,
            mu_1_hat + 4 * sigma_1_hat,
            mu_2_hat + 4 * sigma_2_hat,
            float(threshold) + 10,
        )

        x_grid = np.linspace(
            plot_min,
            plot_max,
            1500,
        )

        fitted_component_1 = (
            p_hat
            *
            norm.pdf(
                x_grid,
                loc=mu_1_hat,
                scale=sigma_1_hat,
            )
        )

        fitted_component_2 = (
            (1.0 - p_hat)
            *
            norm.pdf(
                x_grid,
                loc=mu_2_hat,
                scale=sigma_2_hat,
            )
        )

        fitted_density = (
            fitted_component_1
            +
            fitted_component_2
        )

        fig, ax = plt.subplots(
            figsize=(11, 5.5)
        )

        ax.hist(
            waiting,
            bins=15,
            density=True,
            color="#c9d6ff",
            edgecolor="white",
            alpha=0.90,
            label="Observed waiting times",
        )

        ax.plot(
            x_grid,
            fitted_density,
            color="#4361ee",
            linewidth=3,
            label="Fitted mixture",
        )

        if show_fitted_components:

            ax.plot(
                x_grid,
                fitted_component_1,
                color="#00a896",
                linestyle="--",
                linewidth=2,
                label="Weighted component 1",
            )

            ax.plot(
                x_grid,
                fitted_component_2,
                color="#ff9f1c",
                linestyle="--",
                linewidth=2,
                label="Weighted component 2",
            )

        ax.axvline(
            threshold,
            color="#e63946",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {threshold} min",
        )

        if show_tail_area:

            ax.fill_between(
                x_grid,
                fitted_density,
                where=x_grid >= threshold,
                color="#ff9f1c",
                alpha=0.25,
                label="Displayed tail area",
            )

        ax.set_xlabel(
            "Waiting time (minutes)"
        )

        ax.set_ylabel(
            "Probability density"
        )

        ax.set_title(
            "Fitted normal mixture and observed data"
        )

        ax.legend()

        ax.grid(alpha=0.3)

        show_figure(fig)

        model_probability = (
            normal_mixture_tail_probability(
                threshold,
                mu_1_hat,
                sigma_1_hat,
                mu_2_hat,
                sigma_2_hat,
                p_hat,
            )
        )

        observed_probability = float(
            np.mean(waiting > threshold)
        )

        col_1, col_2 = st.columns(2)

        col_1.metric(
            f"Model: P(X > {threshold})",
            f"{model_probability:.4f}",
            help=(
                "Probability estimated from the "
                "fitted normal mixture."
            ),
        )

        col_2.metric(
            f"Observed proportion above {threshold}",
            f"{observed_probability:.4f}",
            help=(
                "Fraction of observed waiting times "
                "greater than the threshold."
            ),
        )

        st.write(
            f"Model probability: "
            f"**{model_probability * 100:.2f}%**"
        )

        st.write(
            f"Observed proportion: "
            f"**{observed_probability * 100:.2f}%**"
        )

        st.caption(
            """
            The shaded plot is displayed over a finite horizontal
            range. The probability is calculated over the full tail
            from the threshold to infinity.
            """
        )

        st.markdown("---")

        st.subheader(
            "Probability between two waiting times"
        )

        interval = st.slider(
            "Select a waiting-time interval",
            min_value=0,
            max_value=120,
            value=(50, 80),
            step=1,
        )

        lower_interval = interval[0]

        upper_interval = interval[1]

        interval_probability = (
            normal_mixture_interval_probability(
                lower_interval,
                upper_interval,
                mu_1_hat,
                sigma_1_hat,
                mu_2_hat,
                sigma_2_hat,
                p_hat,
            )
        )

        observed_interval = float(
            np.mean(
                (waiting > lower_interval)
                &
                (waiting < upper_interval)
            )
        )

        col_1, col_2 = st.columns(2)

        col_1.metric(
            (
                f"Model: P({lower_interval} < X "
                f"< {upper_interval})"
            ),
            f"{interval_probability:.4f}",
        )

        col_2.metric(
            "Observed proportion in interval",
            f"{observed_interval:.4f}",
        )

        st.markdown("---")

        st.subheader(
            "Model diagnostics"
        )

        # Probability of negative waiting time.
        negative_probability = (
            p_hat
            *
            norm.cdf(
                0,
                loc=mu_1_hat,
                scale=sigma_1_hat,
            )
            +
            (1.0 - p_hat)
            *
            norm.cdf(
                0,
                loc=mu_2_hat,
                scale=sigma_2_hat,
            )
        )

        st.metric(
            "Model probability of negative waiting time",
            f"{negative_probability:.8f}",
        )

        st.caption(
            """
            Normal distributions allow negative values. Waiting time
            cannot be negative, so this probability helps assess
            whether that limitation matters for the fitted model.
            """
        )

        with st.expander(
            "View parameter bounds"
        ):

            bounds_table = pd.DataFrame(
                {
                    "Parameter": [
                        "μ₁",
                        "σ₁",
                        "μ₂",
                        "σ₂",
                        "p",
                    ],
                    "Lower bound": [
                        item[0]
                        for item in bounds
                    ],
                    "Upper bound": [
                        item[1]
                        for item in bounds
                    ],
                }
            )

            st.dataframe(
                bounds_table.style.format(
                    {
                        "Lower bound": "{:.4f}",
                        "Upper bound": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                """
                Bounds on the standard deviations prevent them from
                approaching zero. An unrestricted normal-mixture
                likelihood can become unbounded as a component
                collapses onto an observation.
                """
            )

        with st.expander(
            "Numerical Hessian and eigenvalues"
        ):

            st.write(
                """
                The Hessian is the matrix of second derivatives of
                the negative log-likelihood with respect to the five
                parameters.
                """
            )

            st.latex(
                r"""
                H_{ij}
                =
                \frac{
                    \partial^2[-\ell(\theta)]
                }{
                    \partial\theta_i\partial\theta_j
                }
                """
            )

            if near_boundary:

                st.warning(
                    """
                    At least one estimate is close to a parameter
                    bound. An ordinary unconstrained Hessian check
                    should be interpreted cautiously.
                    """
                )

            try:

                hessian = numerical_hessian(
                    normal_mixture_nll,
                    theta_hat,
                    args=(waiting,),
                )

                if not np.all(
                    np.isfinite(hessian)
                ):

                    raise ValueError(
                        "The Hessian contains non-finite values."
                    )

                labels = [
                    "μ₁",
                    "σ₁",
                    "μ₂",
                    "σ₂",
                    "p",
                ]

                hessian_table = pd.DataFrame(
                    hessian,
                    index=labels,
                    columns=labels,
                )

                st.write(
                    "**Approximate Hessian matrix**"
                )

                st.dataframe(
                    hessian_table.style.format(
                        "{:.4f}"
                    ),
                    use_container_width=True,
                )

                eigenvalues = np.linalg.eigvalsh(
                    hessian
                )

                eigenvalue_table = pd.DataFrame(
                    {
                        "Eigenvalue": eigenvalues
                    }
                )

                st.write(
                    "**Hessian eigenvalues**"
                )

                st.dataframe(
                    eigenvalue_table.style.format(
                        {"Eigenvalue": "{:.6f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if np.all(eigenvalues > 0):

                    st.success(
                        """
                        All computed eigenvalues are positive.
                        This supports a strict local minimum of the
                        negative log-likelihood at an interior
                        stationary point.
                        """
                    )

                else:

                    st.warning(
                        """
                        At least one computed eigenvalue is not
                        positive. Check convergence, boundary effects,
                        and numerical accuracy before drawing a
                        conclusion.
                        """
                    )

                st.caption(
                    """
                    The Hessian is computed numerically. Positive
                    eigenvalues do not by themselves establish a
                    global optimum, and they do not replace checking
                    convergence and parameter bounds.
                    """
                )

            except Exception as error:

                st.warning(
                    "The numerical Hessian could not be calculated."
                )

                st.code(str(error))

        st.markdown("---")

        st.subheader(
            "Download fitted parameters"
        )

        csv_data = parameter_table.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Parameters as CSV",
            data=csv_data,
            file_name="normal_mixture_parameters.csv",
            mime="text/csv",
        )