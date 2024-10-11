# pyrodigy/cli.py
import importlib
import inspect
import json
import os
from datetime import datetime, timedelta

from art import *
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown

from pyrodigy import __version__

# Setup console and paths
# Setup console and paths
console = Console()
README_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "README.md")
)

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
HISTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "history"))
DEFAULT_TTL = timedelta(days=30)

# Ensure history directory exists
os.makedirs(HISTORY_DIR, exist_ok=True)

# Setup Loguru
logger.add("pyrodigy.log", level="DEBUG", rotation="10 MB")


def show_version():
    font = "small  "
    tprint("Pyro's Optimizer CLI", font=font, chr_ignore=True)
    console.print(f"pyrodigy version [bold green]{__version__}[/bold green]")


def list_optimizers():
    if not os.path.isdir(CONFIG_DIR) or not os.path.isdir(DOCS_DIR):
        logger.error("Config or docs directory not found.")
        console.print("[red]Error:[/red] Config or docs directory not found.")
        return

    config_files = {
        f.replace("_config.py", "")
        for f in os.listdir(CONFIG_DIR)
        if f.endswith("_config.py")
    }
    doc_files = {
        os.path.splitext(f)[0] for f in os.listdir(DOCS_DIR) if f.endswith(".md")
    }
    optimizers = sorted(config_files.intersection(doc_files))

    if optimizers:
        logger.info("Available Optimizers:")
        for optimizer in optimizers:
            console.print(f"- {optimizer}")
    else:
        logger.warning("No optimizers found.")
        console.print("No optimizers found.")


def show_optimizer_doc(optimizer_name):
    doc_file = os.path.join(DOCS_DIR, f"{optimizer_name}.md")
    if not os.path.exists(doc_file):
        logger.error(f"Documentation for '{optimizer_name}' not found.")
        console.print("[red]Error:[/red] Documentation not found.")
        return

    with open(doc_file, "r") as f:
        markdown_content = f.read()
    console.print(Markdown(markdown_content))


def load_optimizer_config(optimizer_name):
    config_file = os.path.join(CONFIG_DIR, f"{optimizer_name}_config.py")
    if not os.path.exists(config_file):
        logger.error(f"Configuration for '{optimizer_name}' not found.")
        console.print("[red]Error:[/red] Configuration not found.")
        return None

    spec = importlib.util.spec_from_file_location(optimizer_name, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if hasattr(config_module, "use_case_configs"):
        return config_module.use_case_configs
    else:
        logger.error(f"No use_case_configs found in '{optimizer_name}_config.py'.")
        console.print("[red]Error:[/red] No use_case_configs found.")
        return None


def save_optimizer_config(optimizer_name, configs):
    config_file = os.path.join(CONFIG_DIR, f"{optimizer_name}_config.py")
    with open(config_file, "w") as f:
        f.write(f"use_case_configs = {json.dumps(configs, indent=4)}")
    logger.info(f"Configuration for '{optimizer_name}' saved.")
    console.print("[green]Success:[/green] Configuration updated.")


def show_optimizer_config(optimizer_name):
    configs = load_optimizer_config(optimizer_name)
    if configs:
        console.print_json(data=configs)


def set_optimizer_config(optimizer_name, json_string):
    configs = load_optimizer_config(optimizer_name)
    if not configs:
        return

    try:
        new_config = json.loads(json_string)
    except json.JSONDecodeError:
        logger.error("Invalid JSON string provided.")
        console.print("[red]Error:[/red] Invalid JSON format.")
        return

    configs.update(new_config)
    save_optimizer_config(optimizer_name, configs)


def add_optimizer_config(optimizer_name, config_name, json_string):
    configs = load_optimizer_config(optimizer_name)
    if not configs:
        return

    try:
        new_config = json.loads(json_string)
    except json.JSONDecodeError:
        logger.error("Invalid JSON string provided.")
        console.print("[red]Error:[/red] Invalid JSON format.")
        return

    if config_name in configs:
        console.print(
            f"[red]Error:[/red] Config '{config_name}' already exists. Use 'set' to modify it."
        )
        return

    configs[config_name] = new_config
    save_optimizer_config(optimizer_name, configs)


def rm_optimizer_config(optimizer_name, config_name):
    configs = load_optimizer_config(optimizer_name)
    if not configs:
        return

    if config_name not in configs:
        console.print(f"[red]Error:[/red] Config '{config_name}' not found.")
        return

    del configs[config_name]
    save_optimizer_config(optimizer_name, configs)
    console.print(f"[green]Success:[/green] Config '{config_name}' removed.")


# History management
def record_history(optimizer_name, config_name=None, params=None):
    history_file = os.path.join(HISTORY_DIR, f"{optimizer_name}_history.json")
    history_data = load_history(optimizer_name)

    # Get caller information
    frame = inspect.stack()[2]
    caller_info = {
        "file": frame.filename,
        "line": frame.lineno,
        "function": frame.function,
    }

    # Create history entry
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "optimizer_name": optimizer_name,
        "config_name": config_name,
        "params": params,
        "caller_info": caller_info,
    }

    history_data.append(entry)
    save_history(optimizer_name, history_data)


def load_history(optimizer_name):
    history_file = os.path.join(HISTORY_DIR, f"{optimizer_name}_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []


def save_history(optimizer_name, history_data):
    history_file = os.path.join(HISTORY_DIR, f"{optimizer_name}_history.json")
    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=4)


def show_history(optimizer_name, ttl=DEFAULT_TTL):
    history_data = load_history(optimizer_name)
    cutoff = datetime.utcnow() - ttl
    filtered_data = [
        entry
        for entry in history_data
        if datetime.fromisoformat(entry["timestamp"]) >= cutoff
    ]
    console.print(f"History for '{optimizer_name}' (last {ttl.days} days):")
    for entry in filtered_data:
        console.print(f"- {entry['timestamp']}")


def clear_history(optimizer_name):
    save_history(optimizer_name, [])
    console.print(f"[green]Success:[/green] History for '{optimizer_name}' cleared.")


def apply_ttl(optimizer_name, ttl=DEFAULT_TTL):
    history_data = load_history(optimizer_name)
    cutoff = datetime.utcnow() - ttl
    filtered_data = [
        entry
        for entry in history_data
        if datetime.fromisoformat(entry["timestamp"]) >= cutoff
    ]
    save_history(optimizer_name, filtered_data)
    logger.info(
        f"TTL applied to '{optimizer_name}', keeping entries newer than {ttl.days} days."
    )


# New function to display README with Rich
def show_readme():
    if not os.path.exists(README_PATH):
        logger.error(f"README file not found at {README_PATH}")
        console.print("[red]Error:[/red] README file not found.")
        return

    with open(README_PATH, "r") as readme_file:
        readme_content = readme_file.read()
    console.print(Markdown(readme_content))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pyrodigy Optimizer CLI")
    subparsers = parser.add_subparsers(dest="command")

    # list command
    parser_list = subparsers.add_parser("list", help="List all wrapped optimizers")

    # show command
    parser_show = subparsers.add_parser(
        "show", help="Show documentation for a specific optimizer"
    )
    parser_show.add_argument(
        "optimizer_name", type=str, help="Optimizer to show documentation for"
    )

    # config command
    parser_config = subparsers.add_parser(
        "config", help="Manage optimizer configuration"
    )
    parser_config.add_argument("optimizer_name", type=str, help="Optimizer name")
    parser_config.add_argument(
        "action",
        choices=["get", "set", "add", "rm"],
        help="Get, set, add, or remove a configuration",
    )
    parser_config.add_argument(
        "config_name", nargs="?", help="Name of the config for 'add' or 'rm' action"
    )
    parser_config.add_argument(
        "json_string", nargs="?", help="JSON string for setting or adding configuration"
    )

    parser_history = subparsers.add_parser("history", help="Manage optimizer history")
    parser_history.add_argument("optimizer_name", type=str, help="Optimizer name")
    parser_history.add_argument(
        "action", choices=["show", "clear"], help="Show or clear history"
    )
    parser_history.add_argument(
        "--TTL",
        type=str,
        default="30d",
        help="Time-to-live for history entries (e.g., 30d)",
    )

    # Version command
    parser.add_argument(
        "--version", action="store_true", help="Display the version of pyrodigy"
    )

    # Readme command
    parser_readme = subparsers.add_parser("readme", help="Display the README file")

    args = parser.parse_args()

    if args.version:
        show_version()
    elif args.command == "list":
        list_optimizers()
    elif args.command == "show":
        show_optimizer_doc(args.optimizer_name)
    elif args.command == "config":
        if args.action == "get":
            show_optimizer_config(args.optimizer_name)
        elif args.action == "set" and args.json_string:
            set_optimizer_config(args.optimizer_name, args.json_string)
        elif args.action == "add" and args.config_name and args.json_string:
            add_optimizer_config(
                args.optimizer_name, args.config_name, args.json_string
            )
        elif args.action == "rm" and args.config_name:
            rm_optimizer_config(args.optimizer_name, args.config_name)
        else:
            console.print(
                "[red]Error:[/red] Missing or invalid arguments for config action."
            )
    elif args.command == "history":
        ttl = timedelta(days=int(args.TTL.rstrip("d")))
        if args.action == "show":
            show_history(args.optimizer_name, ttl)
        elif args.action == "clear":
            clear_history(args.optimizer_name)
        apply_ttl(args.optimizer_name, ttl)
    elif args.command == "readme":
        show_readme()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
