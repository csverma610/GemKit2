import unittest
import subprocess
import os
import json
from gemkit.lmdb_storage import LMDBStorage

class TestLmdb2Json(unittest.TestCase):
    """
    Unit tests for the lmdb2json.py script.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        This involves creating a dummy LMDB database.
        """
        self.db_path = "test.lmdb"
        self.json_path = "test.json"
        db = LMDBStorage(self.db_path)
        db.put("key1", "value1")
        db.put("key2", "value2")

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        os.remove(self.db_path)
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def test_script_execution(self):
        """
        Test that the script runs without errors and produces the expected output.
        """
        # Run the script as a subprocess
        result = subprocess.run(
            ["python", "gemkit/lmdb2json.py", self.db_path, self.json_path],
            capture_output=True,
            text=True
        )

        # Check that the script ran successfully
        self.assertEqual(result.returncode, 0)

        # Check that the output contains the expected keys
        self.assertIn("key1", result.stdout)
        self.assertIn("key2", result.stdout)

        # Check that the JSON file was created and contains the expected data
        self.assertTrue(os.path.exists(self.json_path))
        with open(self.json_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["key1"], "value1")
            self.assertEqual(data["key2"], "value2")

if __name__ == '__main__':
    unittest.main()
