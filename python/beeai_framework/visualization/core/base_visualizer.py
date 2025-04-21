#!/usr/bin/env python
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Visualizer

This module provides the abstract base class for all visualization components.
It defines common functionality and configuration options shared across
different visualization types.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import os
import json

logger = logging.getLogger(__name__)


class BaseVisualizer:
    """
    Abstract base class for all visualization components.
    
    This class provides common functionality and configuration options
    for visualization components, including sizing, theming, and export
    capabilities.
    """
    
    def __init__(
        self,
        default_height: int = 600,
        default_width: int = 800,
        theme: str = "light",
        color_scheme: Optional[List[str]] = None
    ):
        """
        Initialize the base visualizer.
        
        Args:
            default_height: Default height for visualizations in pixels
            default_width: Default width for visualizations in pixels
            theme: Visual theme ('light' or 'dark')
            color_scheme: Optional custom color scheme
        """
        self.default_height = default_height
        self.default_width = default_width
        self.theme = theme
        
        # Set default color schemes based on theme
        if color_scheme:
            self.color_scheme = color_scheme
        elif theme == "dark":
            self.color_scheme = [
                "#4C78A8", "#F58518", "#72B7B2", "#54A24B", "#A2719B", 
                "#E45756", "#D67195", "#C9B066", "#91C585"
            ]
        else:  # light theme default
            self.color_scheme = [
                "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
                "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
            ]
    
    def get_theme_colors(self) -> Dict[str, str]:
        """
        Get the theme-specific colors for various elements.
        
        Returns:
            Dictionary of color values for different visualization elements
        """
        if self.theme == "dark":
            return {
                "background": "#282C34",
                "paper_background": "#1E1E1E",
                "text": "#EEEEEE",
                "grid": "#555555",
                "axis": "#888888",
                "highlight": "#F58518",
                "accent": "#4C78A8"
            }
        else:  # light theme
            return {
                "background": "#FFFFFF",
                "paper_background": "#F5F5F5",
                "text": "#333333",
                "grid": "#DDDDDD",
                "axis": "#999999",
                "highlight": "#FF7F0E",
                "accent": "#1F77B4"
            }
    
    def get_layout_defaults(self, height: Optional[int] = None, width: Optional[int] = None) -> Dict[str, Any]:
        """
        Get default layout properties for Plotly visualizations.
        
        Args:
            height: Optional height override
            width: Optional width override
            
        Returns:
            Dictionary with default layout properties
        """
        colors = self.get_theme_colors()
        
        return {
            "height": height or self.default_height,
            "width": width or self.default_width,
            "font": {
                "family": "Arial, sans-serif",
                "color": colors["text"]
            },
            "paper_bgcolor": colors["paper_background"],
            "plot_bgcolor": colors["background"],
            "colorway": self.color_scheme,
            "margin": {"l": 40, "r": 40, "t": 50, "b": 40},
            "xaxis": {
                "gridcolor": colors["grid"],
                "zerolinecolor": colors["grid"]
            },
            "yaxis": {
                "gridcolor": colors["grid"],
                "zerolinecolor": colors["grid"]
            }
        }
    
    def export_figure(self, figure: Any, file_path: str) -> bool:
        """
        Export a visualization figure to a file.
        
        Args:
            figure: Plotly figure to export
            file_path: Path to save the file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Determine file type from extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.html':
                figure.write_html(file_path)
            elif ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(figure.to_dict(), f)
            elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.svg', '.pdf']:
                figure.write_image(file_path)
            else:
                logger.warning(f"Unsupported file extension '{ext}', defaulting to HTML")
                figure.write_html(f"{os.path.splitext(file_path)[0]}.html")
            
            logger.info(f"Visualization exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export visualization: {str(e)}")
            return False 