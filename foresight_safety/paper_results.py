"""Numbers reported in the paper (camera-ready), kept for reference, plots and consistency tests.

Reference values for plots and for the consistency tests.
"""

TABLE_II = {  # aggregate benchmark results, L0/W0/V0, 66 scenarios
    "OpenVLA-oft": {"SSR": 0.42, "USR": 0.06, "SFR": 0.37, "UFR": 0.15, "CC": 0.18, "SASR": 0.35},
    "RDT": {"SSR": 0.30, "USR": 0.10, "SFR": 0.34, "UFR": 0.26, "CC": 0.29, "SASR": 0.22},
    "DP": {"SSR": 0.24, "USR": 0.10, "SFR": 0.34, "UFR": 0.32, "CC": 0.34, "SASR": 0.16},
    "ACT": {"SSR": 0.20, "USR": 0.12, "SFR": 0.31, "UFR": 0.37, "CC": 0.39, "SASR": 0.12},
}

TABLE_III = {  # per-suite breakdown, strongest vs weakest completed baseline
    "OpenVLA-oft": {
        "Force/Torque": {"SSR": 0.46, "CC": 0.17, "SASR": 0.39},
        "Thermal/Energy": {"SSR": 0.35, "CC": 0.26, "SASR": 0.27},
        "Spatial Boundary": {"SSR": 0.39, "CC": 0.22, "SASR": 0.31},
        "Collaborative": {"SSR": 0.44, "CC": 0.16, "SASR": 0.38},
        "Temporal Sequence": {"SSR": 0.47, "CC": 0.14, "SASR": 0.41},
    },
    "ACT": {
        "Force/Torque": {"SSR": 0.22, "CC": 0.40, "SASR": 0.15},
        "Thermal/Energy": {"SSR": 0.12, "CC": 0.54, "SASR": 0.07},
        "Spatial Boundary": {"SSR": 0.16, "CC": 0.47, "SASR": 0.10},
        "Collaborative": {"SSR": 0.23, "CC": 0.34, "SASR": 0.16},
        "Temporal Sequence": {"SSR": 0.27, "CC": 0.29, "SASR": 0.20},
    },
}

TABLE_IV = {  # channel-wise risk exposure of the representative lift_pot unsafe-success episode
    "Force/Torque": {"RET/T": 0.05, "hard_crossed": False},
    "Thermal/Energy": {"RET/T": 0.40, "hard_crossed": False},
    "Spatial Boundary": {"RET/T": 0.60, "hard_crossed": True},
    "Any active risk": {"RET/T": 0.62, "hard_crossed": True},
}

# Derived figures quoted in Sec. V-C (b) and (c).
TEXT_CLAIMS = {
    "success_rate": {"OpenVLA-oft": 0.48},
    "unsafe_share_of_success": {"OpenVLA-oft": 0.125, "RDT": 0.250, "DP": 0.294, "ACT": 0.375},
    "unsafe_share_of_failure": {"OpenVLA-oft": 0.288, "ACT": 0.544},
    "act_ssr_retention": {"Temporal Sequence": 0.57, "Thermal/Energy": 0.34},
    "cc_range": (0.18, 0.39),
    "usr_range": (0.06, 0.12),
    "ufr_range": (0.15, 0.37),
}
