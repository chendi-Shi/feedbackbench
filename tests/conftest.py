"""Avoid this workstation's failing WMI provider before numerical test imports."""

from embodied_research.runtime import configure_platform

configure_platform()
