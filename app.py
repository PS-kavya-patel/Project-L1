import sys
import os

# Seamlessly redirect execution to the new ui folder structure
ui_app_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
with open(ui_app_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execute the real UI file within this context
exec(code, {"__file__": ui_app_path})

