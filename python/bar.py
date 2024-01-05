from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.columns import Columns
import time
import argparse

# Function to get command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Test Script for Rich Progress Bar and Verbose Output")
    parser.add_argument('--progress_bar', action='store_true', help='Enable rich live layout progress bar')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    return args

def update_display(progress, layout, avg_train_loss, avg_val_loss, accuracy, current_lr, cur_sparsity, best_val_loss_str, best_accuracy_str):
    # Update the progress
    layout["progress"].update(progress)

    # Update the current status
    status = format_status(avg_train_loss, avg_val_loss, accuracy, current_lr, cur_sparsity)
    layout["status"].update(status)

    # Update the best results in a table
    results_table = Table.grid(padding=(0, 2))
    results_table.add_column("Metric", justify="left")
    results_table.add_column("Value", justify="left")
    
    results_table.add_row(best_val_loss_str)
    results_table.add_row(best_accuracy_str)
    layout["best_results"].update(results_table)

def format_status(avg_train_loss, avg_val_loss, accuracy, current_lr, cur_sparsity):
    # Format the status information as a single string
    return Columns([
        f"Train Loss: {avg_train_loss:.4f}",
        f"•",
        f"Validation Loss: {avg_val_loss:.4f}",
        f"•",
        f"Validation Accuracy: {accuracy:.2f}%",
        f"•",
        f"Learning Rate: {current_lr:.5f}",
        f"•",
        f"Sparsity: {100*cur_sparsity:.1f}%"
    ], expand=False)

def main():
    args = get_args()

    console = Console()
    if args.verbose:
        console.print("Starting simulated work...")

    if args.progress_bar:
        layout = Layout()
        progress = Progress(TextColumn("[bold blue]Working..."), BarColumn(), TimeRemainingColumn())
        layout.split(
            Layout(name="progress", size=1),
            Layout(name="status", size=2),
            Layout(name="best_results", size=2)
        )
        task_id = progress.add_task("Work", total=10)
        live = Live(layout, console=console, auto_refresh=False)
        live.start()
    else:
        layout = None
        live = None

    for i in range(10):
        time.sleep(0.5)  # Simulate work

        if args.progress_bar:
            progress.update(task_id, advance=1)
            update_display(progress, layout, 0.5+i/10, 0.3+i/10, 85+i/10, 0.001+i/10, 0.05+i/10, "Best Val Loss: 0.3", "Best Acc: 85%")
            live.refresh()

        if args.verbose:
            console.print(f"Step {i+1}/10 completed.")

    if args.progress_bar:
        live.stop()

    if args.verbose:
        console.print("Simulated work completed.")

if __name__ == '__main__':
    main()
