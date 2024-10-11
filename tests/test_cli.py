import json
import os
import unittest
from datetime import datetime, timedelta

from pyrodigy.cli import (
    CONFIG_DIR,
    DOCS_DIR,
    HISTORY_DIR,
    add_optimizer_config,
    apply_ttl,
    clear_history,
    list_optimizers,
    load_history,
    load_optimizer_config,
    record_history,
    rm_optimizer_config,
    set_optimizer_config,
    show_history,
    show_optimizer_config,
    show_optimizer_doc,
)


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure directories exist
        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)

        # Create mock config and doc files
        cls.optimizer_name = "test_optimizer"
        cls.config_path = os.path.join(CONFIG_DIR, f"{cls.optimizer_name}_config.py")
        cls.doc_path = os.path.join(DOCS_DIR, f"{cls.optimizer_name}.md")

        # Sample config and doc content
        config_content = {"default": {"lr": 0.001, "beta": 0.9}}
        with open(cls.config_path, "w") as f:
            f.write(f"use_case_configs = {json.dumps(config_content)}")

        doc_content = "# Test Optimizer Documentation"
        with open(cls.doc_path, "w") as f:
            f.write(doc_content)

    def test_list_optimizers(self):
        list_optimizers()
        # Manually check console output or use rich console capturing for verification

    def test_show_optimizer_doc(self):
        show_optimizer_doc(self.optimizer_name)
        # Manually check console output or capture the printed output

    def test_load_and_show_optimizer_config(self):
        configs = load_optimizer_config(self.optimizer_name)
        self.assertIsNotNone(configs)
        self.assertIn("default", configs)

        show_optimizer_config(self.optimizer_name)

        # Manually check console output or capture printed JSON

    def test_set_optimizer_config(self):
        new_config = json.dumps({"default": {"lr": 0.01}})
        set_optimizer_config(self.optimizer_name, new_config)

        configs = load_optimizer_config(self.optimizer_name)
        self.assertEqual(configs["default"]["lr"], 0.01)

    def test_add_optimizer_config(self):
        new_config = json.dumps({"new_config": {"lr": 0.02}})
        add_optimizer_config(self.optimizer_name, "new_config", new_config)

        configs = load_optimizer_config(self.optimizer_name)
        self.assertIn("new_config", configs)

    def test_rm_optimizer_config(self):
        rm_optimizer_config(self.optimizer_name, "default")

        configs = load_optimizer_config(self.optimizer_name)
        self.assertNotIn("default", configs)

    def test_record_and_show_history(self):
        record_history(self.optimizer_name, "default", {"lr": 0.001, "beta": 0.9})

        history = load_history(self.optimizer_name)
        self.assertTrue(len(history) > 0)

        show_history(self.optimizer_name)
        # Manually check console output or capture the printed history

    def test_clear_history(self):
        record_history(self.optimizer_name, "default", {"lr": 0.001})

        clear_history(self.optimizer_name)
        history = load_history(self.optimizer_name)
        self.assertEqual(len(history), 0)

    def test_apply_ttl(self):
        record_history(self.optimizer_name, "default", {"lr": 0.001})

        # Simulate an old entry
        old_entry = {
            "timestamp": (datetime.utcnow() - timedelta(days=60)).isoformat(),
            "optimizer_name": self.optimizer_name,
            "config_name": "default",
            "params": {"lr": 0.001},
        }
        history_file = os.path.join(HISTORY_DIR, f"{self.optimizer_name}_history.json")
        with open(history_file, "w") as f:
            json.dump([old_entry], f)

        apply_ttl(self.optimizer_name, timedelta(days=30))
        history = load_history(self.optimizer_name)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
