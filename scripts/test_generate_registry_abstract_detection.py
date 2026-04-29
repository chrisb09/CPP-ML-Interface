import importlib.util
import pathlib
import unittest


_module_path = pathlib.Path(__file__).resolve().with_name("generate_registry.py")
_spec = importlib.util.spec_from_file_location("generate_registry", _module_path)
generate_registry = importlib.util.module_from_spec(_spec) # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(generate_registry)


class TestAbstractDetection(unittest.TestCase):
    def setUp(self):
        try:
            import clang.cindex as cindex
        except ModuleNotFoundError:
            self.skipTest("clang.cindex is not available in this Python environment")

        self.cindex = cindex
        self.cursor_kind = cindex.CursorKind

        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.parse_args = ['-std=c++17', '-I.', '-Iinclude', '-xc++']

    def _get_class_node(self, header_rel_path: str, class_name: str):
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
                return node, str(header_path)

        self.fail(f"Class {class_name} not found in {header_rel_path}")

    def test_abstract_detection_matches_ground_truth(self):
        cases = [
            ("include/normalization/ml_coupling_normalization.hpp", "MLCouplingNormalization", True),
            ("include/normalization/ml_coupling_minmax_normalization.hpp", "MLCouplingMinMaxNormalization", False),
            ("include/provider/ml_coupling_provider.hpp", "MLCouplingProvider", True),
            ("include/behavior/ml_coupling_behavior.hpp", "MLCouplingBehavior", True),
            ("include/behavior/ml_coupling_behavior_default.hpp", "MLCouplingBehaviorDefault", False),
            ("include/application/ml_coupling_application.hpp", "MLCouplingApplication", True),
            ("include/application/ml_coupling_application_turbulence_closure.hpp", "MLCouplingApplicationTurbulenceClosure", False),
        ]

        mismatches = []
        for header_rel_path, class_name, expected_abstract in cases:
            node, header_abs_path = self._get_class_node(header_rel_path, class_name)

            try:
                clang_detected_abstract = bool(node.is_abstract_record())
            except Exception:
                clang_detected_abstract = False

            fallback_detected_abstract = generate_registry.class_body_contains_pure_virtual(node, header_abs_path)
            effective_detected_abstract = generate_registry.is_effectively_abstract(node, header_abs_path)

            if effective_detected_abstract != expected_abstract:
                mismatches.append(
                    f"{class_name}: expected={expected_abstract}, "
                    f"clang={clang_detected_abstract}, fallback={fallback_detected_abstract}, "
                    f"effective={effective_detected_abstract}"
                )

        if mismatches:
            self.fail("\n".join(mismatches))


if __name__ == "__main__":
    unittest.main()
