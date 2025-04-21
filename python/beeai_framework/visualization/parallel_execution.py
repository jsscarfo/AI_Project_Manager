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

import asyncio
import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import numpy as np

from beeai_framework.workflows.parallel.controller import (
    ParallelExecutionController, ParallelTask, TaskState
)
from beeai_framework.visualization.base import BaseVisualization

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionStats:
    """Statistics for task execution"""
    task_id: str
    name: str
    state: TaskState
    dependencies: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelExecutionVisualizer(BaseVisualization):
    """
    Visualization tools for parallel execution monitoring.
    
    Features:
    - Task dependency graph visualization
    - Gantt chart for task execution timeline
    - Resource utilization charts
    - Performance metrics visualization
    """
    
    def __init__(self, controller: Optional[ParallelExecutionController] = None):
        """Initialize with an optional controller to monitor"""
        self.controller = controller
        self.task_stats: Dict[str, TaskExecutionStats] = {}
        self._resource_usage_history: Dict[str, List[Tuple[float, float]]] = {}
        self._task_state_history: Dict[str, List[Tuple[float, TaskState]]] = {}
        
        # Register for task events if controller is provided
        if self.controller:
            self._register_event_handlers()
    
    def set_controller(self, controller: ParallelExecutionController) -> None:
        """Set the controller to monitor"""
        self.controller = controller
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register for task events from the controller"""
        if not self.controller:
            return
            
        self.controller.on_task_event("task_scheduled", self._handle_task_scheduled)
        self.controller.on_task_event("task_started", self._handle_task_started)
        self.controller.on_task_event("task_completed", self._handle_task_completed)
        self.controller.on_task_event("task_failed", self._handle_task_failed)
    
    def _handle_task_scheduled(self, event: Any) -> None:
        """Handle task scheduled event"""
        task = event.task
        self.task_stats[task.id] = TaskExecutionStats(
            task_id=task.id,
            name=task.name,
            state=task.state,
            dependencies=task.dependencies,
            metadata=task.metadata
        )
        
        # Initialize state history
        self._task_state_history[task.id] = [(time.time(), task.state)]
    
    def _handle_task_started(self, event: Any) -> None:
        """Handle task started event"""
        task = event.task
        if task.id in self.task_stats:
            stats = self.task_stats[task.id]
            stats.state = task.state
            stats.start_time = task.started_at or time.time()
            
            # Update state history
            self._task_state_history[task.id].append((time.time(), task.state))
            
            # Initialize resource usage tracking
            for constraint in task.resource_constraints:
                resource_name = constraint.name
                resource_amount = constraint.amount
                
                if resource_name not in self._resource_usage_history:
                    self._resource_usage_history[resource_name] = []
                
                self._resource_usage_history[resource_name].append((time.time(), resource_amount))
                
                if resource_name not in stats.resource_usage:
                    stats.resource_usage[resource_name] = resource_amount
    
    def _handle_task_completed(self, event: Any) -> None:
        """Handle task completed event"""
        task = event.task
        if task.id in self.task_stats:
            stats = self.task_stats[task.id]
            stats.state = task.state
            stats.end_time = task.completed_at or time.time()
            
            if stats.start_time:
                stats.duration = stats.end_time - stats.start_time
            
            # Update state history
            self._task_state_history[task.id].append((time.time(), task.state))
            
            # Update resource usage tracking - resources released
            for resource_name in stats.resource_usage:
                if resource_name in self._resource_usage_history:
                    self._resource_usage_history[resource_name].append((time.time(), 0.0))
    
    def _handle_task_failed(self, event: Any) -> None:
        """Handle task failed event"""
        task = event.task
        if task.id in self.task_stats:
            stats = self.task_stats[task.id]
            stats.state = task.state
            stats.end_time = task.completed_at or time.time()
            
            if stats.start_time:
                stats.duration = stats.end_time - stats.start_time
            
            # Update state history
            self._task_state_history[task.id].append((time.time(), task.state))
            
            # Update resource usage tracking - resources released
            for resource_name in stats.resource_usage:
                if resource_name in self._resource_usage_history:
                    self._resource_usage_history[resource_name].append((time.time(), 0.0))
    
    def visualize_dependency_graph(self, filename: Optional[str] = None) -> None:
        """
        Visualize the task dependency graph.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self.controller or not self.task_stats:
            logger.warning("No controller or task data available for visualization")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # State to color mapping
        state_colors = {
            TaskState.PENDING: '#aaaaaa',    # Gray
            TaskState.READY: '#3498db',      # Blue
            TaskState.RUNNING: '#f39c12',    # Orange
            TaskState.COMPLETED: '#2ecc71',  # Green
            TaskState.FAILED: '#e74c3c',     # Red
            TaskState.CANCELLED: '#95a5a6'   # Light gray
        }
        
        # Add nodes
        for task_id, stats in self.task_stats.items():
            G.add_node(
                task_id, 
                label=stats.name,
                state=stats.state.value,
                color=state_colors.get(stats.state, '#000000')
            )
        
        # Add edges
        for task_id, stats in self.task_stats.items():
            for dep_id in stats.dependencies:
                if dep_id in self.task_stats:
                    G.add_edge(dep_id, task_id)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Create node colors list
        node_colors = [G.nodes[n]['color'] for n in G.nodes]
        
        # Use hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Draw the graph
        nx.draw(
            G, 
            pos, 
            with_labels=True,
            node_color=node_colors,
            edge_color='#bbbbbb',
            font_size=10,
            font_weight='bold',
            node_size=3000,
            arrows=True
        )
        
        # Add legend
        legend_elements = [
            Patch(facecolor=color, label=state.value)
            for state, color in state_colors.items()
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        plt.title('Task Dependency Graph', fontsize=16)
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def visualize_timeline(self, filename: Optional[str] = None) -> None:
        """
        Create a Gantt chart visualization of task execution timeline.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self.task_stats:
            logger.warning("No task data available for timeline visualization")
            return
        
        # Filter tasks with start times
        tasks_with_timing = {
            task_id: stats for task_id, stats in self.task_stats.items()
            if stats.start_time is not None
        }
        
        if not tasks_with_timing:
            logger.warning("No tasks with timing information available")
            return
        
        # State to color mapping
        state_colors = {
            TaskState.RUNNING: '#f39c12',    # Orange
            TaskState.COMPLETED: '#2ecc71',  # Green
            TaskState.FAILED: '#e74c3c',     # Red
            TaskState.CANCELLED: '#95a5a6'   # Light gray
        }
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Process task data for Gantt chart
        task_names = []
        start_times = []
        durations = []
        colors = []
        now = time.time()
        
        # Sort tasks by start time
        sorted_tasks = sorted(
            tasks_with_timing.values(),
            key=lambda x: x.start_time or now
        )
        
        for i, stats in enumerate(sorted_tasks):
            task_names.append(f"{stats.name} ({stats.task_id})")
            
            start_time = datetime.fromtimestamp(stats.start_time)
            start_times.append(start_time)
            
            if stats.end_time:
                duration = (stats.end_time - stats.start_time) / 3600  # Convert to hours
            else:
                # Task still running
                duration = (now - stats.start_time) / 3600  # Convert to hours
            
            durations.append(duration)
            colors.append(state_colors.get(stats.state, '#aaaaaa'))
        
        # Create horizontal bars
        y_positions = range(len(task_names))
        ax.barh(y_positions, durations, left=start_times, color=colors, height=0.5)
        
        # Configure y-axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels(task_names)
        
        # Configure x-axis for time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add grid lines for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add legend
        legend_elements = [
            Patch(facecolor=color, label=state.value)
            for state, color in state_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Tasks')
        ax.set_title('Parallel Task Execution Timeline', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def visualize_resource_usage(self, filename: Optional[str] = None) -> None:
        """
        Visualize resource usage over time.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self._resource_usage_history:
            logger.warning("No resource usage data available for visualization")
            return
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Colors for different resources
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Plot each resource usage over time
        for i, (resource_name, usage_data) in enumerate(self._resource_usage_history.items()):
            if not usage_data:
                continue
                
            # Extract times and usage values
            times = [datetime.fromtimestamp(t) for t, _ in usage_data]
            usage = [u for _, u in usage_data]
            
            # Calculate cumulative usage at each time point
            cumulative_usage = []
            current_usage = 0
            
            for u in usage:
                if u > 0:
                    current_usage += u
                else:
                    # A zero value means resource was released
                    current_usage -= usage_data[len(cumulative_usage)-1][1]
                
                cumulative_usage.append(max(0, current_usage))
            
            # Plot resource usage
            color = colors[i % len(colors)]
            ax.plot(times, cumulative_usage, label=resource_name, color=color, linewidth=2)
            ax.fill_between(times, 0, cumulative_usage, color=color, alpha=0.3)
        
        # Configure x-axis for time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add grid lines
        ax.grid(linestyle='--', alpha=0.7)
        
        # Add legend, labels and title
        ax.legend(loc='upper left')
        ax.set_xlabel('Time')
        ax.set_ylabel('Resource Usage')
        ax.set_title('Resource Utilization Over Time', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def visualize_task_state_transitions(self, filename: Optional[str] = None) -> None:
        """
        Visualize task state transitions over time.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self._task_state_history:
            logger.warning("No task state history available for visualization")
            return
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # State to numeric value mapping for plotting
        state_values = {
            TaskState.PENDING: 0,
            TaskState.READY: 1,
            TaskState.RUNNING: 2,
            TaskState.COMPLETED: 3,
            TaskState.FAILED: 4,
            TaskState.CANCELLED: 5
        }
        
        # State to color mapping
        state_colors = {
            TaskState.PENDING: '#aaaaaa',    # Gray
            TaskState.READY: '#3498db',      # Blue
            TaskState.RUNNING: '#f39c12',    # Orange
            TaskState.COMPLETED: '#2ecc71',  # Green
            TaskState.FAILED: '#e74c3c',     # Red
            TaskState.CANCELLED: '#95a5a6'   # Light gray
        }
        
        # Plot state transitions for each task
        for task_id, state_history in self._task_state_history.items():
            if task_id not in self.task_stats or not state_history:
                continue
                
            # Get task name for label
            task_name = self.task_stats[task_id].name
            
            # Extract times and states
            times = [datetime.fromtimestamp(t) for t, _ in state_history]
            states = [state_values.get(s, 0) for _, s in state_history]
            
            # Get final state for color
            final_state = state_history[-1][1]
            color = state_colors.get(final_state, '#000000')
            
            # Plot state transitions
            ax.plot(times, states, label=f"{task_name} ({task_id})", color=color, marker='o')
        
        # Configure y-axis with state labels
        ax.set_yticks(list(state_values.values()))
        ax.set_yticklabels([s.value for s in state_values.keys()])
        
        # Configure x-axis for time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add grid lines
        ax.grid(linestyle='--', alpha=0.7)
        
        # Add legend, labels and title
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Task State')
        ax.set_title('Task State Transitions', fontsize=16)
        
        # Adjust layout
        plt.subplots_adjust(right=0.7)  # Make room for the legend
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def visualize_performance_metrics(self, filename: Optional[str] = None) -> None:
        """
        Visualize performance metrics for parallel execution.
        
        Args:
            filename: Optional filename to save the visualization
        """
        if not self.task_stats:
            logger.warning("No task data available for performance metrics")
            return
        
        # Collect metrics
        completed_tasks = {
            task_id: stats for task_id, stats in self.task_stats.items()
            if stats.state == TaskState.COMPLETED and stats.duration is not None
        }
        
        if not completed_tasks:
            logger.warning("No completed tasks with duration information")
            return
        
        # Set up the plot (2x2 grid)
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Task Duration Distribution
        durations = [stats.duration for stats in completed_tasks.values()]
        axs[0, 0].hist(durations, bins=20, color='#3498db', alpha=0.7)
        axs[0, 0].set_xlabel('Duration (seconds)')
        axs[0, 0].set_ylabel('Number of Tasks')
        axs[0, 0].set_title('Task Duration Distribution')
        axs[0, 0].grid(linestyle='--', alpha=0.7)
        
        # 2. Task State Counts
        state_counts = {}
        for stats in self.task_stats.values():
            if stats.state not in state_counts:
                state_counts[stats.state] = 0
            state_counts[stats.state] += 1
        
        states = list(state_counts.keys())
        counts = list(state_counts.values())
        
        # State to color mapping
        state_colors = {
            TaskState.PENDING: '#aaaaaa',    # Gray
            TaskState.READY: '#3498db',      # Blue
            TaskState.RUNNING: '#f39c12',    # Orange
            TaskState.COMPLETED: '#2ecc71',  # Green
            TaskState.FAILED: '#e74c3c',     # Red
            TaskState.CANCELLED: '#95a5a6'   # Light gray
        }
        
        colors = [state_colors.get(state, '#000000') for state in states]
        
        axs[0, 1].bar(
            [s.value for s in states], 
            counts,
            color=colors,
            alpha=0.7
        )
        axs[0, 1].set_xlabel('Task State')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].set_title('Task State Distribution')
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Resource Utilization Boxplot
        if self._resource_usage_history:
            resource_names = list(self._resource_usage_history.keys())
            resource_data = []
            
            for resource_name in resource_names:
                resource_data.append([u for _, u in self._resource_usage_history[resource_name]])
            
            axs[1, 0].boxplot(resource_data, labels=resource_names)
            axs[1, 0].set_xlabel('Resource')
            axs[1, 0].set_ylabel('Usage')
            axs[1, 0].set_title('Resource Usage Distribution')
            axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        else:
            axs[1, 0].text(0.5, 0.5, 'No resource usage data', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[1, 0].transAxes)
        
        # 4. Dependency Chain Length Analysis
        dependency_lengths = {}
        
        def calc_chain_length(task_id: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0  # Avoid cycles
            
            visited.add(task_id)
            
            if task_id not in self.task_stats:
                return 0
            
            stats = self.task_stats[task_id]
            
            if not stats.dependencies:
                return 0
            
            max_length = 0
            for dep_id in stats.dependencies:
                length = calc_chain_length(dep_id, visited.copy()) + 1
                max_length = max(max_length, length)
            
            return max_length
        
        for task_id in self.task_stats:
            dependency_lengths[task_id] = calc_chain_length(task_id)
        
        # Calculate duration by dependency length
        dep_lengths = sorted(set(dependency_lengths.values()))
        durations_by_length = {length: [] for length in dep_lengths}
        
        for task_id, length in dependency_lengths.items():
            if task_id in completed_tasks:
                durations_by_length[length].append(completed_tasks[task_id].duration)
        
        # Calculate average durations
        avg_durations = []
        std_durations = []
        for length in dep_lengths:
            if durations_by_length[length]:
                avg_durations.append(np.mean(durations_by_length[length]))
                std_durations.append(np.std(durations_by_length[length]))
            else:
                avg_durations.append(0)
                std_durations.append(0)
        
        # Plot dependency length vs duration
        axs[1, 1].bar(dep_lengths, avg_durations, yerr=std_durations, 
                       color='#9b59b6', alpha=0.7)
        axs[1, 1].set_xlabel('Dependency Chain Length')
        axs[1, 1].set_ylabel('Average Duration (seconds)')
        axs[1, 1].set_title('Task Duration by Dependency Chain Length')
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def export_metrics(self, filename: str) -> None:
        """
        Export performance metrics to a file.
        
        Args:
            filename: Filename to save the metrics
        """
        import json
        from datetime import datetime
        
        metrics = {
            "export_time": datetime.now().isoformat(),
            "tasks": {},
            "resource_usage": {},
            "summary": {
                "total_tasks": len(self.task_stats),
                "state_counts": {},
                "avg_duration": None,
                "max_duration": None
            }
        }
        
        # Task metrics
        for task_id, stats in self.task_stats.items():
            metrics["tasks"][task_id] = {
                "name": stats.name,
                "state": stats.state.value,
                "dependencies": stats.dependencies,
                "start_time": stats.start_time,
                "end_time": stats.end_time,
                "duration": stats.duration,
                "resource_usage": stats.resource_usage
            }
        
        # Resource usage history (simplified)
        for resource_name, usage_data in self._resource_usage_history.items():
            metrics["resource_usage"][resource_name] = [
                {"time": t, "usage": u} for t, u in usage_data
            ]
        
        # Summary stats
        state_counts = {}
        durations = []
        
        for stats in self.task_stats.values():
            if stats.state not in state_counts:
                state_counts[stats.state.value] = 0
            state_counts[stats.state.value] += 1
            
            if stats.duration is not None:
                durations.append(stats.duration)
        
        metrics["summary"]["state_counts"] = state_counts
        
        if durations:
            metrics["summary"]["avg_duration"] = sum(durations) / len(durations)
            metrics["summary"]["max_duration"] = max(durations)
            metrics["summary"]["min_duration"] = min(durations)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {filename}")
    
    def visualize_all(self, base_filename: str = "parallel_execution") -> None:
        """
        Generate all visualizations and save them to files.
        
        Args:
            base_filename: Base filename to use for all visualizations
        """
        self.visualize_dependency_graph(f"{base_filename}_dependencies.png")
        self.visualize_timeline(f"{base_filename}_timeline.png")
        self.visualize_resource_usage(f"{base_filename}_resources.png") 
        self.visualize_task_state_transitions(f"{base_filename}_states.png")
        self.visualize_performance_metrics(f"{base_filename}_performance.png")
        self.export_metrics(f"{base_filename}_metrics.json") 