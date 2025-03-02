
import json

def load_command_logic(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

command_logic = load_command_logic("command_logic.json")

def list_commands(mode):
    current_mode_commands = command_logic.get(mode, {})
    available_commands = "
".join(current_mode_commands.keys())
    return f"Available commands in {mode} mode: {available_commands}"
