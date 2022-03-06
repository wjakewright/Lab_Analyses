""" Module to analyze and summarize the main metrics of lever press behavior
    for a single mouse across all sessions. Stores output as a dataclass."""

from dataclasses import dataclass
import numpy as np

from Lab_Analyses.Behavior import summarize_lever_behavior as slb


def analyze_mouse_lever_behavior(files, sessions):
    """Function to analyze all the """



#-------------------------------------------------------------------------
#---------------------------DATACLASSES USED------------------------------
#-------------------------------------------------------------------------

@dataclass
class Mouse_Lever_Data:
    """Dataclass for storing processed lever press behavior data across
       all sessions for a single mouse"""
    
    mouse_id: str
