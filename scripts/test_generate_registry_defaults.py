import importlib.util
import pathlib
import unittest


_module_path = pathlib.Path(__file__).resolve().with_name("generate_registry.py")
_spec = importlib.util.spec_from_file_location("generate_registry", _module_path)
generate_registry = importlib.util.module_from_spec(_spec) # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(generate_registry)


class TestDefaultTokenExtraction(unittest.TestCase):
    def test_simple_identifier_default(self):
        tokens = [
            "std", "::", "function", "<", "bool", "(", "int", ")", ">",
            "prohibit_inference", "=", "allow_inference_at_all_steps", ")"
        ]
        value = generate_registry._extract_default_from_param_tokens(tokens)
        self.assertEqual(value, "allow_inference_at_all_steps")

    def test_lambda_with_capture_and_commas(self):
        tokens = [
            "std", "::", "function", "<", "bool", "(", "int", ")", ">", "f", "=",
            "[", "a", ",", "b", "]",
            "(", "int", "step", ")",
            "{", "return", "step", "<", "a", ",", "step", "<", "b", ";", "}",
            ",",
            "next_param"
        ]
        value = generate_registry._extract_default_from_param_tokens(tokens)
        self.assertEqual(
            value,
            "[ a , b ] ( int step ) { return step < a , step < b ; }"
        )

    def test_lambda_stops_at_top_level_right_paren(self):
        tokens = [
            "std", "::", "function", "<", "bool", "(", "int", ")", ">", "f", "=",
            "[", "]", "(", "int", "s", ")", "{", "return", "s", "%", "2", "==", "0", ";", "}",
            ")"
        ]
        value = generate_registry._extract_default_from_param_tokens(tokens)
        self.assertEqual(value, "[ ] ( int s ) { return s % 2 == 0 ; }")

    def test_top_level_template_brace_default(self):
        tokens = [
            "std", "::", "array", "<", "int", ",", "3", ">", "arr", "=",
            "std", "::", "array", "<", "int", ",", "3", ">", "{", "}",
            ",",
            "next_param"
        ]
        value = generate_registry._extract_default_from_param_tokens(tokens)
        self.assertEqual(value, "std :: array < int , 3 > { }")

    def test_nested_template_brace_default(self):
        tokens = [
            "std", "::", "map", "<", "int", ",", "std", "::", "vector", "<", "int", ">", ">",
            "m", "=",
            "std", "::", "map", "<", "int", ",", "std", "::", "vector", "<", "int", ">", ">", "{", "}",
            ",",
            "next_param"
        ]
        value = generate_registry._extract_default_from_param_tokens(tokens)
        self.assertEqual(value, "std :: map < int , std :: vector < int > > { }")


if __name__ == "__main__":
    unittest.main()
