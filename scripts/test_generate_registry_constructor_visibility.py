import importlib.util
import pathlib
import unittest


_module_path = pathlib.Path(__file__).resolve().with_name("generate_registry.py")
_spec = importlib.util.spec_from_file_location("generate_registry", _module_path)
generate_registry = importlib.util.module_from_spec(_spec) # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(generate_registry)


class TestConstructorVisibility(unittest.TestCase):
    def setUp(self):
        try:
            import clang.cindex as cindex
        except ModuleNotFoundError:
            self.skipTest("clang.cindex is not available in this Python environment")

        self.cindex = cindex
        self.cursor_kind = cindex.CursorKind
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.parse_args = ['-std=c++17', '-I.', '-Iinclude', '-xc++']

    def _find_class_node(self, header_rel_path: str, class_name: str):
        header_path = self.repo_root / header_rel_path
        index = self.cindex.Index.create()
        tu = index.parse(str(header_path), args=self.parse_args)

        for node in tu.cursor.walk_preorder():
            if node.kind not in (
                self.cursor_kind.CLASS_DECL,
                self.cursor_kind.STRUCT_DECL,
                self.cursor_kind.CLASS_TEMPLATE,
            ):
                continue

            if node.spelling != class_name:
                continue

            node_file = str(node.location.file) if node.location and node.location.file else ""
            if node_file and (node_file.endswith(header_rel_path) or str(header_path).endswith(node_file)):
                return node

        self.fail(f"Class {class_name} not found in {header_rel_path}")

    def test_smartsim_private_ctor_not_exposed(self):
        node = self._find_class_node(
            "include/provider/ml_coupling_provider_smartsim.hpp",
            "MLCouplingProviderSmartsim",
        )

        constructors = generate_registry.get_class_constructors(node, "MLCouplingProviderSmartsim")

        self.assertEqual(
            len(constructors),
            1,
            f"Expected exactly one public constructor, got {len(constructors)}: {constructors}",
        )

        public_ctor_param_names = [name for _, name, _ in constructors[0]]
        self.assertNotIn("hosts", public_ctor_param_names)
        self.assertNotIn("ports", public_ctor_param_names)


if __name__ == "__main__":
    unittest.main()
